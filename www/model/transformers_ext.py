import math

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, KLDivLoss, Softmax, BCEWithLogitsLoss, BCELoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import RobertaForMultipleChoice, BertForMultipleChoice, BertModel, RobertaModel, DebertaModel, DebertaPreTrainedModel
from transformers.models.deberta.modeling_deberta import ContextPooler, StableDropout
from transformers.activations import ACT2FN, gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers import RobertaConfig

class DebertaForMultipleChoice(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaModel(config)
        # self.pooler = ContextPooler(config)
        # output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(config.hidden_size, 1)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)        

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # pooled_output = outputs[1]

        seqout = outputs[0]
        cls = seqout[:,:1,:]
        cls = cls/math.sqrt(seqout.size(-1))
        att_score = torch.matmul(cls, seqout.transpose(-1,-2))
        att_mask = attention_mask.unsqueeze(1).to(att_score)
        att_score = att_mask*att_score + (att_mask-1)*10000.0
        att_score = torch.nn.functional.softmax(att_score, dim=-1)
        pool = torch.matmul(att_score, seqout).squeeze(-2)
        cls = self.dropout(pool)
        logits = self.classifier(cls).float().squeeze(-1)
        reshaped_logits = logits.view([-1, num_choices])
        loss = 0

        # pooled_output = self.pooler(encoder_layer)

        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# RoBERTa classification head borrowed from HuggingFace
class ClassificationHead(nn.Module):

    def __init__(self, config, input_all_tokens=True):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        drop_out = getattr(config, "cls_dropout", None)
        if drop_out is None:
          drop_out = getattr(config, "dropout_rate", None)
        if drop_out is None:
          drop_out = getattr(config, "hidden_dropout_prob", None)
        assert drop_out is not None, "Didn't set dropout!"
        self.dropout = nn.Dropout(drop_out)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels
        self.input_all_tokens = input_all_tokens

    def forward(self, features, return_embeddings=False):
        if self.input_all_tokens:
          x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
          x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        if return_embeddings:
          emb = x
        x = self.dropout(x)
        x = self.out_proj(x)
        if not return_embeddings:
          return x
        else:
          return x, emb

# Tiered model proposed for TRIP
class TieredModelPipeline(nn.Module):
  def __init__(self, embedding, num_sents, num_attributes, labels_per_att, config_class, model_name, device, ablation=[], loss_weights=[0.0, 0.4, 0.4, 0.2, 0.0]): # labels_per_att is a dictionary mapping attribute index to number of labels
    super().__init__()

    # Embedding and dropout
    self.embedding = embedding
    drop_out = getattr(embedding.config, "cls_dropout", None)
    if drop_out is None:
      drop_out = getattr(embedding.config, "dropout_rate", None)
    if drop_out is None:
      drop_out = getattr(embedding.config, "hidden_dropout_prob", None)
    assert drop_out is not None, "Didn't set dropout!"
    self.dropout = nn.Dropout(drop_out)   

    # State classifiers
    self.num_attributes = num_attributes
    self.labels_per_att = labels_per_att
    self.num_state_labels = sum(list(labels_per_att.values()))

    self.attribute_classifier = nn.Linear(embedding.config.hidden_size, num_attributes)

    config = config_class.from_pretrained(model_name)
    self.precondition_classifiers = []
    self.effect_classifiers = []
    for i in range(num_attributes):
      config.num_labels = labels_per_att[i]
      self.precondition_classifiers.append(ClassificationHead(config, input_all_tokens=False).to(device))
      self.effect_classifiers.append(ClassificationHead(config, input_all_tokens=False).to(device))
    
    # Conflict detector components
    embedding_proj_size = 256

    encoding_size = 0
    if 'embeddings' not in ablation:
      encoding_size += embedding_proj_size
    if 'states' not in ablation or 'states-attention' in ablation:
      self.states_size = 0
      if 'states-labels' in ablation:
        self.states_size += 2*num_attributes
      if 'states-logits' in ablation:
        self.states_size += 2*self.num_state_labels

      if 'embeddings' in ablation:
        self.states_repeat = 1
      else:
        # Want states to take up similar size of input to embeddings
        self.states_repeat = embedding_proj_size // (self.states_size)
        assert self.states_repeat > 0

      # encoding_size += (self.states_size) * self.states_repeat
      if 'states' not in ablation:
        encoding_size += embedding_proj_size

    # Project to same size
    if 'embeddings' not in ablation:
      self.embedding_proj = nn.Linear(embedding.config.hidden_size, embedding_proj_size)
    if 'states' not in ablation:
      self.states_proj = nn.Linear(self.states_size, embedding_proj_size)
    if 'states-attention' in ablation:
      self.states_attention = nn.Sequential(nn.Linear(self.states_size, embedding_proj_size), nn.Sigmoid()) # Attention mechanism conditioned on states

    n_head = 8
    encoding_pad_zeros = 0
    if encoding_size % n_head != 0:
      encoding_pad_zeros = n_head - (encoding_size % n_head)
    encoding_size += encoding_pad_zeros
    self.encoding_size = encoding_size
    self.encoding_pad_zeros = encoding_pad_zeros
    assert self.encoding_size % n_head == 0, "Conflict detector encoding size (%s) should be divisible by n_heads=%s" % (str(self.encoding_size), str(n_heads))
    transformer_layers = nn.TransformerEncoderLayer(d_model=encoding_size, nhead=n_head, dim_feedforward=4*encoding_size, dropout=drop_out, activation='relu')
    self.detector = nn.TransformerEncoder(transformer_layers, num_layers=6)
    self.num_sents = num_sents
    num_spans = (num_sents) * (num_sents-1) // 2
    self.num_spans = num_spans
    self.decoder = nn.Linear(num_sents * encoding_size, num_sents) # pivot point multilabel classification layer

    self.ablation = ablation
    self.loss_weights = loss_weights


  def forward(self, input_ids, input_lengths, input_entities, attention_mask=None, token_type_ids=None, attributes=None, preconditions=None, effects=None, conflicts=None, labels=None, training=False):

    batch_size, num_stories, num_entities, num_sents, seq_length = input_ids.shape
    assert num_stories == 2
    assert num_sents == self.num_sents

    input_lengths = input_lengths.view(-1)
    input_entities = input_entities.view(-1)
    
    length_mask = torch.ones((batch_size * num_stories * num_entities, num_sents), requires_grad=False).to(input_lengths.device)
    for i in range(self.num_sents):
      length_mask[input_lengths <= i, i] = 0 # Use input lengths to zero out state and conflict preds wherever there isn't a sentence
    length_mask = length_mask.view(batch_size*num_stories, num_entities, num_sents)
    for i in range(num_entities):
      length_mask[input_entities <= i, i, :] = 0 # Use input entity counts to zero out state and conflict preds wherever there isn't an entity
    length_mask = length_mask.view(batch_size * num_stories * num_entities, num_sents)

    # 1) Embed the inputs
    if token_type_ids is not None:
      print(token_type_ids)
      print(token_type_ids.shape)
      out = self.embedding(
              input_ids.view(batch_size * num_stories * num_entities * num_sents, -1).long(),
              attention_mask=attention_mask.view(batch_size * num_stories * num_entities * num_sents, -1) if attention_mask is not None else None,
              token_type_ids=token_type_ids.view(batch_size * num_stories * num_entities * num_sents, -1),
              output_hidden_states=False)
    else:
      out = self.embedding(
              input_ids.view(batch_size * num_stories * num_entities * num_sents, -1).long(),
              attention_mask=attention_mask.view(batch_size * num_stories * num_entities * num_sents, -1) if attention_mask is not None else None,
              output_hidden_states=False)
      
    if len(out[0].shape) < 3:
      out[0] = out[0].unsqueeze(0)
    out = out[0][:,0,:] # entity-sentence embeddings


    # 2) State classification
    return_dict = {}

    loss_attributes = None
    if 'attributes' not in self.ablation:
      # 2a) Attribute classification
      out_a = self.attribute_classifier(out)
      out_a = torch.sigmoid(out_a)
      return_dict['out_attributes'] = out_a # Extract normalized logits (will turn to preds later)

      if attributes is not None:
        loss_fct = BCEWithLogitsLoss()
        loss_attributes = loss_fct(out_a, attributes.view(batch_size * num_stories * num_entities * num_sents, -1).float())
        return_dict['loss_attributes'] = loss_attributes

    else:
      out_a = out


    # 2b) Precondition classification
    loss_preconditions = None
    loss_fct = CrossEntropyLoss()
    if preconditions is not None:
      loss_preconditions = 0.0

    out_preconditions = torch.zeros((batch_size * num_stories * num_entities * num_sents, self.num_attributes)).to(self.embedding.device)
    out_preconditions_softmax = torch.tensor([]).to(self.embedding.device)
    for i in range(self.num_attributes):
      out_s = self.precondition_classifiers[i](out, return_embeddings=False)

      with torch.no_grad(): # Don't allow backprop from conflict detection to state classifiers
        out_preconditions_softmax = torch.cat((out_preconditions_softmax, out_s), dim=-1)

      if 'attributes' not in self.ablation:
        # If attribute classifier predicted 0, zero out positive classes and vice versa
        out_s[:, 0] *= (1 - out_a[:, i])
        out_s[:, 1:] *= out_a[:, i].repeat(self.labels_per_att[i]-1,1).t()

      out_preconditions[:, i] = torch.argmax(out_s, dim=1) # Extract predicted value

      if preconditions is not None:
        loss_preconditions += loss_fct(out_s.view(-1, self.precondition_classifiers[i].num_labels), preconditions[:, :, :, :, i].view(-1))

    out_preconditions *= length_mask.view(-1).repeat(self.num_attributes, 1).t() # Mask out any nonexistent entities or sentences
    assert length_mask.view(-1).shape[0] == out_preconditions.shape[0]
    return_dict['out_preconditions'] = out_preconditions # * length_mask.view(-1).repeat(self.num_attributes, 1).t()
    if preconditions is not None:
      return_dict['loss_preconditions'] = loss_preconditions


    # 2c) Effect classification
    loss_effects = None
    loss_fct = CrossEntropyLoss()
    if effects is not None:
      loss_effects = 0.0

    out_effects = torch.zeros((batch_size * num_stories * num_entities * num_sents, self.num_attributes)).to(self.embedding.device)
    out_effects_softmax = torch.tensor([]).to(self.embedding.device)
    for i in range(self.num_attributes):
      out_s = self.effect_classifiers[i](out, return_embeddings=False)
      with torch.no_grad(): # Don't allow backprop from conflict detection to state classifiers
        out_effects_softmax = torch.cat((out_effects_softmax, out_s), dim=-1)

      if 'attributes' not in self.ablation:
        # If attribute classifier predicted 0, zero out positive classes and vice versa
        out_s[:, 0] *= (1 - out_a[:, i])
        out_s[:, 1:] *= out_a[:, i].repeat(self.labels_per_att[i]-1,1).t()

      out_effects[:, i] = torch.argmax(out_s, dim=1) # Extract predicted value

      if effects is not None:
        loss_effects += loss_fct(out_s.view(-1, self.effect_classifiers[i].num_labels), effects[:, :, :, :, i].view(-1))
    
    out_effects *= length_mask.view(-1).repeat(self.num_attributes, 1).t() # Mask out any nonexistent entities or sentences
    assert length_mask.view(-1).shape[0] == out_effects.shape[0]      
    return_dict['out_effects'] = out_effects # * length_mask.view(-1).repeat(self.num_attributes, 1).t()
    if effects is not None:
      return_dict['loss_effects'] = loss_effects


    # 3) Conflict detection
    if training and 'states-teacher-forcing' in self.ablation:
      out_preconditions = preconditions.view(batch_size * num_stories * num_entities * num_sents, self.num_attributes).float()
      out_effects = effects.view(batch_size * num_stories * num_entities * num_sents, self.num_attributes).float()

    if 'states-labels' not in self.ablation and 'states-logits' in self.ablation:
      out_preconditions = out_preconditions_softmax
      out_effects = out_effects_softmax
    elif 'states-labels' in self.ablation and 'states-logits' in self.ablation:
      out_preconditions = torch.cat((out_preconditions, out_preconditions_softmax), dim=-1)
      out_effects = torch.cat((out_effects, out_effects_softmax), dim=-1)

    # Project states and embeddings to same size
    if 'embeddings' not in self.ablation:
      out = self.dropout(out)
      out = self.embedding_proj(out)

    out_states = torch.cat((out_preconditions, out_effects), dim=-1)
    if 'states' not in self.ablation:
      out_states = self.dropout(out_states)
      out_states = self.states_proj(out_states)
    if 'states-attention' in self.ablation:
      out *= self.states_attention(out_states)

    # Concatenate entity-sentence embedding to preconditions and effects
    if 'states' in self.ablation and 'embeddings' not in self.ablation:
      out = out
    elif 'states' not in self.ablation and 'embeddings' in self.ablation:
      out = out_states
    else:
      out = torch.cat((out, out_states), dim=-1)

    # Pad with a few zeros
    out = torch.cat((out, torch.zeros(out.shape[:-1] + (self.encoding_pad_zeros,)).to(self.embedding.device)), dim=-1)
    out = out.view(batch_size * num_stories * num_entities, num_sents, -1)

    # Run through transformer
    out = self.dropout(out)
    out = self.detector(out.transpose(0, 1)).transpose(0, 1)
    out = out.reshape(batch_size * num_stories * num_entities, -1)

    # Then through output projection - get logits for belief on each span to be conflicting
    out = self.dropout(out)
    out = self.decoder(out)
    out = torch.sigmoid(out) * length_mask # Do sigmoid again since this happened inside loss function
    assert input_lengths.shape[0] == out.shape[0]
    return_dict['out_conflicts'] = out # * length_mask

    loss_conflicts = None
    if conflicts is not None:
      loss_fct = BCELoss()
      loss_conflicts = loss_fct(out, conflicts.view(batch_size*num_stories*num_entities, -1).float())

      return_dict['loss_conflicts'] = loss_conflicts

    # 4) Story choice classification
    out = out.view(batch_size, num_stories, num_entities, -1) # Reshape to one prediction per example-story-entity triples    
    out = -torch.sum(out, dim=(2,3)) / 2 # divide by 2 so the expected sum is 1 for conflicting story (2 conflicting sentences)
    return_dict['out_stories'] = out
    loss_stories = None
    if labels is not None:
      loss_fct = CrossEntropyLoss()
      loss_stories = loss_fct(out, labels)
      return_dict['loss_stories'] = loss_stories

    total_loss = 0.0
    if loss_attributes is not None: # Not incorporated
      total_loss += self.loss_weights[0] * loss_attributes
    if loss_preconditions is not None:
      total_loss += self.loss_weights[1] * loss_preconditions / self.num_attributes
    if loss_effects is not None:
      total_loss += self.loss_weights[2] * loss_effects / self.num_attributes
    if loss_conflicts is not None:
      total_loss += self.loss_weights[3] * loss_conflicts
    if loss_stories is not None:
      total_loss += self.loss_weights[4] * loss_stories
    if loss_attributes is None and loss_preconditions is None and loss_effects is None and loss_conflicts is None and loss_stories is None:
      total_loss = None

    if total_loss is not None:
      return_dict['total_loss'] = total_loss

    return return_dict
