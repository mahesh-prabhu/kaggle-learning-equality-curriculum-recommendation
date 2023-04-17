import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import fbeta_score
from l_utils import *
import os

def load_checkpoint(filepath, device):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model = model.to(device)
    model.eval()
    return model

def binary_acc(y_pred, y_test, add_sigmoid = False):
    if (add_sigmoid):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
    else:
        y_pred_tag = y_pred

    correct_results_sum = torch.sum(y_pred_tag == y_test)
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    # print(y_pred_tag[0:10])
    # print(y_test[0:10])
    # print(f'{acc.item()}   ... {correct_results_sum}  / {y_test.shape[0]}')
    
    return acc

#
# Below is from how to use minilm from sentence-transformers website
#
def mean_pooling_with_token_embs(token_embeddings, attention_mask):
    """
    Average the output embeddings using the attention mask 
    to ignore certain tokens.
    """
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

# Below from https://www.kaggle.com/code/nbroad/multiple-negatives-ranking-loss-lecr/notebook#Model-and-Loss
def cos_sim(a, b):
    # From https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py#L31
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

# Basically the same as this: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py
class MultipleNegativesRankingLoss(torch.nn.Module):
    
    def __init__(self, scaling_factor = 20.0):
        super().__init__()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.scaling_factor = scaling_factor

    def forward(self, embeddings_a, embeddings_b, labels=None):
        """
        Compute similarity between `a` and `b`.
        Labels have the index of the row number at each row. 
        This indicates that `a_i` and `b_j` have high similarity 
        when `i==j` and low similarity when `i!=j`.
        """

        similarity_scores = (
            cos_sim(embeddings_a, embeddings_b) * float(self.scaling_factor)
        )  # Not too sure why to scale it by 20: https://github.com/UKPLab/sentence-transformers/blob/b86eec31cf0a102ad786ba1ff31bfeb4998d3ca5/sentence_transformers/losses/MultipleNegativesRankingLoss.py#L57
        
        labels = torch.tensor(
            range(len(similarity_scores)),
            dtype=torch.long,
            device=similarity_scores.device,
        )  # Example a[i] should match with b[i]

        return self.loss_function(similarity_scores, labels)

class MultipleNegativesRankingArcosLoss(torch.nn.Module):
    
    def __init__(self, loss_function = None, margin = 0.4, scaling_factor = 20.0, margin_increase_per_epoch = 0.0, train_batches_per_epoch = 0):
        super().__init__()
        if loss_function is None:
            self.loss_function = torch.nn.CrossEntropyLoss()
        else:
            self.loss_function = loss_function
        self.margin = margin
        self.scaling_factor = scaling_factor
        self.margin_increase_per_epoch = margin_increase_per_epoch
        self.train_batches_per_epoch = train_batches_per_epoch
        self.batch_count = 0

    def forward(self, embeddings_a, embeddings_b, debug = False):
        # Step 1: Get cos-similarity 
        cos_sim_scores = (
            cos_sim(embeddings_a, embeddings_b)
        )
        
        # this prevents nan when a value slightly crosses 1.0 due to numerical error
        cos_sim_scores = cos_sim_scores.clip(-1+1e-7, 1-1e-7)

        if debug:
            print(cos_sim_scores.size())
        
        # Step 3: get the angle from the cosine value
        arcosine = cos_sim_scores.arccos()

        m = torch.nn.functional.one_hot(torch.arange(embeddings_a.size(0)), num_classes = embeddings_a.size(0)).to(cos_sim_scores.device) * self.margin # We are assuming embeddings_a and embeddings_b are tensors of same size, which should be batch-size * embeddings-size.

        if debug:
            print(m)
        
        # Step 4: add margin
        arcosine += m

        if debug:
            print(arcosine)
        
        # Step 5:
        cos_sim_scores = arcosine.cos() * float(self.scaling_factor)

        if debug:
            print(cos_sim_scores)

        labels = torch.tensor(
            range(len(cos_sim_scores)),
            dtype=torch.long,
            device=cos_sim_scores.device,
        )  # Example a[i] should match with b[i]

        if debug:
            print(labels)

        # Linearly increase margin at every call to the loss function
        if self.train_batches_per_epoch > 0 and self.margin_increase_per_epoch != 0:
            self.batch_count += 1
            if self.batch_count == self.train_batches_per_epoch:
                self.margin += self.margin_increase_per_epoch
                self.batch_count = 0
        
        # Step 6:
        return self.loss_function(cos_sim_scores, labels)

class ArcosLossWithWeights(torch.nn.Module):
    
    def __init__(self, emb_size, output_classes, loss_function = None, margin = 0.4, scaling_factor = 20.0, margin_increase_per_epoch = 0.0, train_batches_per_epoch = 0):
        super().__init__()
        self.W = torch.nn.Parameter(torch.Tensor(output_classes, emb_size))
        self.output_classes = output_classes
        torch.nn.init.kaiming_uniform_(self.W)
        if loss_function is None:
            self.loss_function = torch.nn.CrossEntropyLoss()
        else:
            self.loss_function = loss_function
        self.margin = margin
        self.scaling_factor = scaling_factor
        self.margin_increase_per_epoch = margin_increase_per_epoch
        self.train_batches_per_epoch = train_batches_per_epoch
        self.batch_count = 0

    def forward(self, embeddings, labels, debug = False):
        # Step 1: Get cos-similarity 
        cos_sim_scores = (
            cos_sim(embeddings, self.W)
        )
        
        # this prevents nan when a value slightly crosses 1.0 due to numerical error
        cos_sim_scores = cos_sim_scores.clip(-1+1e-7, 1-1e-7)

        if debug:
            print(cos_sim_scores.size())
        
        # Step 3: get the angle from the cosine value
        arcosine = cos_sim_scores.arccos()

        m = torch.nn.functional.one_hot(labels, num_classes = self.output_classes).to(cos_sim_scores.device) * self.margin

        if debug:
            print(m)
        
        # Step 4: add margin to the angle
        arcosine += m

        if debug:
            print(arcosine)
        
        # Step 5: compute the cosine from the angle
        cos_sim_scores = arcosine.cos() * float(self.scaling_factor)

        if debug:
            print(cos_sim_scores)

        # Increase margin at the end of an epoch
        if self.train_batches_per_epoch > 0 and self.margin_increase_per_epoch != 0:
            self.batch_count += 1
            if self.batch_count == self.train_batches_per_epoch:
                self.margin += self.margin_increase_per_epoch
                self.batch_count = 0
        
        # Step 6:
        return self.loss_function(cos_sim_scores, labels)
    
def arcface_loss(cosine, targ, m=.4):
    # this prevents nan when a value slightly crosses 1.0 due to numerical error
    cosine = cosine.clip(-1+1e-7, 1-1e-7) 
    # Step 3:
    arcosine = cosine.arccos()
    # Step 4:
    arcosine += F.one_hot(targ, num_classes = output_classes) * m
    # Step 5:
    cosine2 = arcosine.cos()
    # Step 6:
    return F.cross_entropy(cosine2, targ)
    
def calc_f2_score(y_pred, y_true):
    y_pred = y_pred.numpy().astype('bool')
    y_true = y_true.numpy().astype('bool')
    #return fbeta_score(np.array(y_true), np.array(y_pred), beta=2, average = 'samples')
    f2 = fbeta_score(y_true, y_pred, beta=2, average = 'binary')
    f1 = fbeta_score(y_true, y_pred, beta=1, average = 'binary')
    return (f1, f2)

def make_pred( t_emb, c_emb, threshold):
    # simi = torch.diagonal(cos_sim(t_emb, c_emb), 0).unsqueeze(dim=1) ## **todo: I didn't use scaling factor here!! Do I need to?
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    simi = cos(t_emb, c_emb)
    return (simi > threshold)

def compute_val_scores(t_model, c_model, data_loader, threshold):
    t_model.eval()
    c_model.eval()
    val_acc = 0
    f2_score = 0
    f1_score = 0
    with torch.no_grad():
        for topic, content, label in tqdm(data_loader, desc="Validation"):
            topic   = topic.to(device)
            content = content.to(device)
            label   = label.to(device)

            t_emb = t_model(topic)
            c_emb = c_model(content)

            pred = make_pred(t_emb, c_emb, threshold)

            label = (label == 1.0).squeeze() # convert label to boolean
            # print(label.shape)
            # print(pred.shape)
            acc  = binary_acc(pred,label, add_sigmoid = False)
            val_acc  += acc.item()
            f1, f2 = calc_f2_score(pred.cpu(), label.cpu())
            f2_score += f2
            f1_score += f1
    t_model.train()
    c_model.train()
    n = len(data_loader)
    return val_acc/n, f1_score/n, f2_score/n



# https://stackoverflow.com/questions/67870579/rowwise-numpy-isin-for-2d-arrays
# def np_rowise_intersection(A, B):
#     m = (A[:,:,None] == B[:,None,:]).any(-1)
#     return m
def np_rowise_intersection(A, B):
    return np.array([np.isin(a, b) for a,b in zip(A, B)])

def compute_recall_at_k(t_model, c_model, device, topics_features, contents_features, corr_data, k, batch_size):
    # Steps:
    # 1. for each batch of topics compute pairwaise distances between topic and content embeddings
    # 2. get top k results for each topic
    # 3. calculate recall


    # Let's convert our correlation data into a numpy array
    # https://stackoverflow.com/questions/10346336/list-of-lists-into-numpy-array
    corr_data_l_l = list(corr_data.to_numpy()) # Convert into a list of list, some of the lists might be of different sizes
    max_length = max(map(len, corr_data_l_l)) # find the maximum list size
    corr_data_np = np.array([xi+[-1]*(max_length-len(xi)) for xi in corr_data_l_l]) # Add -1 into rows that have less than max_length
    
    recall = 0
    t_model.eval()
    c_model.eval()
    with torch.no_grad():
        start_idx = 0
        c_emb = c_model(contents_features.to(device))
        for i in tqdm(range(0, ceildiv(topics_features.size(0),batch_size))):

            
            start_idx = i * batch_size
            end_idx   = (i+1) * batch_size
            if (end_idx > topics_features.size(0)):
                end_idx = topics_features.size(0)
            topics_features_batch   = topics_features[start_idx:end_idx]
            t_emb_batch = t_model(topics_features_batch.to(device))

            # Step 1
            simi = cos_sim(t_emb_batch, c_emb)

            # Step 2
            matched_content_idxs = torch.topk(simi, dim = 1, k = k, largest = True).indices.cpu().detach().numpy()

            # Step 3
            corr_data_np_batch = corr_data_np[start_idx:end_idx]
            per_row_TP_count = np_rowise_intersection(matched_content_idxs, corr_data_np_batch).sum(axis=1) # Count of True Positives (TP) per row

            per_row_TP_FP_count = np.sum(np.array(corr_data_np_batch) >= 0, axis=1) # we have plugged in -1 in the corr_data_np array to make sure the row sizes match up. So we ignore those to get a count of TP+FP.
            
            batch_recall = (per_row_TP_count/per_row_TP_FP_count).sum()
            recall += batch_recall

    t_model.train()
    c_model.train()
    n = topics_features.size(0)
    return recall/n


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)

# compute_c_embs_for_val -> compute_embs_from_model
def compute_embs_from_model(model, batch_size, features, disable_tqdm = True):
    """
    Compute the embs for all the data
    """

    num_rows = features.size(0)
    num_batches = ceildiv(num_rows,batch_size)
    embs = []
    for i in tqdm(range(0, num_batches), disable=disable_tqdm):
        start_idx = i * batch_size
        end_idx   = (i+1) * batch_size
        if (end_idx > num_rows):
            end_idx = num_rows
        features_batch = features[start_idx:end_idx]
        embs.append(model(features_batch))
    embs = torch.cat(embs, axis = 0)
    return embs


    
def get_metric_vals_from_scores(scores, metric, metric_key):
    metric_vals = []
    for score in scores:
        v = score[metric]
        if (metric_key is not None):
            v = v[metric_key]
        metric_vals.append(v)
    return metric_vals

def get_max(l):
    cur_max = None
    cur_max_i = None
    for i, v in enumerate(l):
        if (cur_max is None) or (v > cur_max) :
            cur_max = v
            cur_max_i = i
    return cur_max, cur_max_i

def save_best_model(model_dict, old_model_filename, model_filename_prefix, model_path, save_model_fn,
                    train_scores, val_scores, metric, metric_key,  dry_run):
    metric_vals = get_metric_vals_from_scores(val_scores, metric, metric_key)
    # get current metric val
    cur_metric_val = metric_vals[len(metric_vals)-1]
    # get previous best metric val
    best_metric_val, _ = get_max(metric_vals[:len(metric_vals)-1])
    # check if current score is better than best score
    if (best_metric_val is None) or (cur_metric_val > best_metric_val):
        if metric_key is None:
            metric_key = ''
        # delete the old_model and save new one
        if old_model_filename is not None:
            if dry_run:
                pt(f'Dry run for deleting {old_model_filename}')
            else:
                pt(f'Deleting {old_model_filename}')
                os.remove(old_model_filename)
        epoch_count = len(val_scores)-1
        model_filename = model_path + f'{model_filename_prefix}_epoch_{epoch_count}_{metric}_{metric_key}_{cur_metric_val}.pth'
        if dry_run:
            pt(f'Dry run for saving {model_filename}')
        else:
            pt(f'Saving {model_filename}')
            save_model_fn(model_dict, model_filename)
    else:
        model_filename = old_model_filename

    return model_filename

def check_improvement(scores, metric, metric_key, patience):
    metric_vals = get_metric_vals_from_scores(scores, metric, metric_key)
    # get the best metric val
    _, best_metric_index = get_max(metric_vals)
    # if the gap between the best metric val and current epoch is greater than patience, then return 0 else return 1
    if len(metric_vals) - (best_metric_index+1) > patience:
        return 0
    return 1 

def compute_recall_at_k_for_two_trans(t_model, c_model, device, topics_features, contents_features, corr_data, k_vals, batch_size):
    # Steps:
    # 1. for each batch of topics compute pairwaise distances between topic and content embeddings
    # 2. get top k results for each topic
    # 3. calculate recall


    # Let's convert our correlation data into a numpy array
    # https://stackoverflow.com/questions/10346336/list-of-lists-into-numpy-array
    corr_data_l_l = list(corr_data.to_numpy()) # Convert into a list of list, some of the lists might be of different sizes
    max_length = max(map(len, corr_data_l_l)) # find the maximum list size
    corr_data_np = np.array([xi+[-1]*(max_length-len(xi)) for xi in corr_data_l_l]) # Add -1 into rows that have less than max_length
    
    recalls = [0] * len(k_vals)
    t_model.eval()
    c_model.eval()
    with torch.no_grad():
        start_idx = 0
        # c_emb = c_model(contents_features.to(device))
        c_emb = compute_embs_from_model(c_model, batch_size, contents_features.to(device))
        for i in tqdm(range(0, ceildiv(topics_features.size(0),batch_size))):

            
            start_idx = i * batch_size
            end_idx   = (i+1) * batch_size
            if (end_idx > topics_features.size(0)):
                end_idx = topics_features.size(0)
            topics_features_batch   = topics_features[start_idx:end_idx]
            t_emb_batch = t_model(topics_features_batch.to(device))

            # Step 1
            simi = cos_sim(t_emb_batch, c_emb)

            # Step 2
            matched_content_idxs_list = [torch.topk(simi, dim = 1, k = k, largest = True).indices.cpu().detach().numpy() for k in k_vals]

            # Step 3
            corr_data_np_batch = corr_data_np[start_idx:end_idx]
            per_row_TP_count_list = [np_rowise_intersection(matched_content_idxs, corr_data_np_batch).sum(axis=1) for matched_content_idxs in matched_content_idxs_list] # Count of True Positives (TP) per row

            per_row_TP_FN_count = np.sum(np.array(corr_data_np_batch) >= 0, axis=1) # we have plugged in -1 in the corr_data_np array to make sure the row sizes match up. So we ignore those to get a count of TP+FP.
            
            batch_recall_list = [(per_row_TP_count/per_row_TP_FN_count).sum() for per_row_TP_count in per_row_TP_count_list]
            recalls = [(recall+batch_recall) for (recall, batch_recall) in zip(recalls, batch_recall_list)]

    t_model.train()
    c_model.train()
    n = topics_features.size(0)
    recalls = [round(recall/n, 3) for recall in recalls]
    return dict(zip(k_vals, recalls))


def compute_c_embs_for_val_for_one_trans(trans_model, c_model, batch_size, features, content_sep_indices):
    """
    Compute the c_embs for all the content at once
    """

    num_rows = features.size(0)
    num_batches = ceildiv(num_rows,batch_size)
    c_embs = []
    for i in range(0, num_batches):
        start_idx = i * batch_size
        end_idx   = (i+1) * batch_size
        if (end_idx > num_rows):
            end_idx = num_rows
        features_batch = features[start_idx:end_idx]
        c_embs.append(c_model(trans_model(features_batch, content_sep_indices)))
    c_embs_for_val = torch.cat(c_embs, axis = 0)
    return c_embs_for_val

def compute_recall_at_k_for_one_trans_two_towers(trans_model, t_model, c_model, device, topics_features, topic_sep_indices, contents_features, content_sep_indices, corr_data, k_vals, batch_size):
    # Steps:
    # 1. for each batch of topics compute pairwaise distances between topic and content embeddings
    # 2. get top k results for each topic
    # 3. calculate recall


    # Let's convert our correlation data into a numpy array
    # https://stackoverflow.com/questions/10346336/list-of-lists-into-numpy-array
    corr_data_l_l = list(corr_data.to_numpy()) # Convert into a list of list, some of the lists might be of different sizes
    max_length = max(map(len, corr_data_l_l)) # find the maximum list size
    corr_data_np = np.array([xi+[-1]*(max_length-len(xi)) for xi in corr_data_l_l]) # Add -1 into rows that have less than max_length
    
    recalls = [0] * len(k_vals)
    trans_model.eval()
    t_model.eval()
    c_model.eval()
    with torch.no_grad():
        start_idx = 0
        # c_emb = c_model(contents_features.to(device))
        c_emb = compute_c_embs_for_val_for_one_trans(trans_model, c_model, batch_size, contents_features.to(device), content_sep_indices.to(device))
        for i in tqdm(range(0, ceildiv(topics_features.size(0),batch_size))):

            
            start_idx = i * batch_size
            end_idx   = (i+1) * batch_size
            if (end_idx > topics_features.size(0)):
                end_idx = topics_features.size(0)
            topics_features_batch   = topics_features[start_idx:end_idx]
            topic_sep_indices_batch = topic_sep_indices[start_idx:end_idx]
            t_emb_batch = t_model(trans_model(topics_features_batch.to(device), topic_sep_indices_batch.to(device)))

            # Step 1
            simi = cos_sim(t_emb_batch, c_emb)

            # Step 2
            matched_content_idxs_list = [torch.topk(simi, dim = 1, k = k, largest = True).indices.cpu().detach().numpy() for k in k_vals]

            # Step 3
            corr_data_np_batch = corr_data_np[start_idx:end_idx]
            per_row_TP_count_list = [np_rowise_intersection(matched_content_idxs, corr_data_np_batch).sum(axis=1) for matched_content_idxs in matched_content_idxs_list] # Count of True Positives (TP) per row

            per_row_TP_FN_count = np.sum(np.array(corr_data_np_batch) >= 0, axis=1) # we have plugged in -1 in the corr_data_np array to make sure the row sizes match up. So we ignore those to get a count of TP+FP.
            
            batch_recall_list = [(per_row_TP_count/per_row_TP_FN_count).sum() for per_row_TP_count in per_row_TP_count_list]
            recalls = [(recall+batch_recall) for (recall, batch_recall) in zip(recalls, batch_recall_list)]

    trans_model.train()
    t_model.train()
    c_model.train()
    n = topics_features.size(0)
    recalls = [round(recall/n, 3) for recall in recalls]
    return dict(zip(k_vals, recalls))


def save_model(model_dict, filename):
    checkpoint = {}

    for k, v in model_dict.items():
        checkpoint[k] = v.state_dict()
        
    torch.save(checkpoint, filename)


def load_model(model_dict, filename):
    checkpoint = torch.load(filename)
    for k, v in model_dict.items():
        v.load_state_dict(checkpoint[k])
    return model_dict


def calculate_mean_average_recall_at_k(matched_content_idxs_list, k_vals, corr_data, fill_value = -1):
    # Let's convert our correlation data into a numpy array
    # https://stackoverflow.com/questions/10346336/list-of-lists-into-numpy-array
    corr_data_l_l = list(corr_data.to_numpy()) # Convert into a list of list, some of the lists might be of different sizes
    max_length = max(map(len, corr_data_l_l)) # find the maximum list size
    corr_data_np = np.array([xi+[fill_value]*(max_length-len(xi)) for xi in corr_data_l_l]) # Add -1 into rows that have less than max_length

    # Count of True Positives (TP) per row/topic
    per_row_TP_count_list = [np_rowise_intersection(matched_content_idxs, corr_data_np).sum(axis=1) for matched_content_idxs in matched_content_idxs_list] 

    # Count TP + FN per row/topic
    per_row_TP_FN_count = np.sum(np.array(corr_data_np) >= 0, axis=1) # we have plugged in -1 in the corr_data_np array to make sure the row sizes match up. So we ignore those to get a count of TP+FP.

    # Calculate recall per row/topic and sum it up
    recalls = [(per_row_TP_count/per_row_TP_FN_count).sum() for per_row_TP_count in per_row_TP_count_list]

    total_topics = corr_data_np.shape[0]
    recalls = [round(recall/total_topics, 3) for recall in recalls]
    return dict(zip(k_vals, recalls))

def compute_top_k_predictions(t_model, device, topics_features, contents_features, k_vals, batch_size):
    # Steps:
    # 1. for each batch of topics compute pairwaise distances between topic and content embeddings
    # 2. get top k results for each topic
    # 3. calculate recall

    t_model.eval()
    with torch.no_grad():
        start_idx = 0
        pt('Computing c_embs ... ')
        c_emb = compute_embs_from_model(t_model, batch_size, contents_features.to(device), disable_tqdm = False)
        all_matched_content_idxs = [None]*len(k_vals)
        pt('Computing top-k predictions ...')
        for i in tqdm(range(0, ceildiv(topics_features.size(0),batch_size))):
        # for i in tqdm(range(topics_features_l[0].size(0))):
            
            start_idx = i * batch_size
            end_idx   = (i+1) * batch_size
            if (end_idx > topics_features.size(0)):
                end_idx = topics_features.size(0)
            # start_idx = i
            # end_idx = i+1
            t_emb_batch = t_model(topics_features[start_idx:end_idx].to(device))

            # Step 1
            simi = cos_sim(t_emb_batch, c_emb)
            #print(simi.size())

            # Step 2
            matched_content_idxs_list = [torch.topk(simi, dim = 1, k = k, largest = True).indices.cpu().detach().numpy() for k in k_vals]
            if (all_matched_content_idxs[0] is None):
                all_matched_content_idxs = matched_content_idxs_list
            else:
                all_matched_content_idxs = [np.vstack((a1,a2)) for a1,a2 in zip(all_matched_content_idxs, matched_content_idxs_list)]
            #print(all_matched_content_idxs[0].shape)

    return all_matched_content_idxs

def compute_top_k_predictions_with_lang_filter(t_model, device, topics_features, contents_features, topics_df, contents_df, k_vals, batch_size, fill_value):
    # Steps:
    # 1. for each batch of topics compute pairwaise distances between topic and content embeddings
    # 2. get top k results for each topic

    t_model.eval()
    with torch.no_grad():
        all_matched_content_idxs = None
        pt('Computing content embeddings ...')
        c_embs = compute_embs_from_model(t_model, batch_size, contents_features.to(device), disable_tqdm = False)
        pt('Computing topic embeddings ...')
        t_embs = compute_embs_from_model(t_model, batch_size, topics_features.to(device), disable_tqdm = False)
        pt('Computing top-k for each topic ...')
        for t_idx in tqdm(range(0, topics_features.size(0))):
            t_lang = topics_df.loc[t_idx, 'language']
            c_idxs = contents_df.query(f'language=="{t_lang}"').index.tolist()

            # Step 1
            simi = cos_sim(t_embs[t_idx:t_idx+1,:], c_embs[c_idxs,:])

            simi = simi.squeeze(0)

            # Step 2
            num_contents = simi.size()[0]
            matched_content_idxs_list = [torch.topk(simi, dim = 0, k = min(k,num_contents), largest = True).indices.cpu().detach().numpy() for k in k_vals] # This incides are for our subset array defined by c_idxs
            matched_content_idxs_list = [np.array(c_idxs)[matched_content_idxs] for matched_content_idxs in matched_content_idxs_list ] # Map the indices back to the original array
            if (all_matched_content_idxs is None):
                all_matched_content_idxs = [[l] for l in matched_content_idxs_list]
            else:
                for i, l in enumerate(matched_content_idxs_list):
                    all_matched_content_idxs[i].append(l)

    all_matched_content_idxs_np_list = []
    for l in all_matched_content_idxs:
        df = pd.DataFrame(l)
        print(df.shape)
        all_matched_content_idxs_np_list.append(df.fillna(fill_value).values)
    return all_matched_content_idxs_np_list

def get_top_k_preds_for_models( trans_model_names, model_path, model_filenames, TransModel, device, topics_features_l, contents_features_l, k_vals, batch_size, corr_data, topics_df_l, contents_df_l):
    model_preds = []
    # The reason we use FILL_VALUE = -2, is because we take a rowise_intersection later to get the True-positives, and the other 2d array has -1 as fill value. So we don't want to have a false intersection with same fill values, so we use -2 as fill value here instead of -1.
    FILL_VALUE = -2
    for i, (trans_model_name, model_filename) in enumerate(zip(trans_model_names, model_filenames)):
        pt(f'Loading model {model_filename} ... ')
        trans_model = TransModel(trans_model_name = trans_model_name)
        load_model( {'trans_model': trans_model}, filename = model_path+model_filename)
        trans_model = trans_model.to(device)
        preds = compute_top_k_predictions_with_lang_filter(trans_model,
                                                           device,
                                                           topics_features = topics_features_l[i],
                                                           contents_features = contents_features_l[i],
                                                           topics_df = topics_df_l[i],
                                                           contents_df = contents_df_l[i],
                                                           k_vals = k_vals,
                                                           batch_size = batch_size,
                                                           fill_value = FILL_VALUE
                                                           )
        pt('Garbage collection ... ')
        del trans_model
        report_gpu()
        model_preds.append(preds)
        recalls = calculate_mean_average_recall_at_k(preds, k_vals, corr_data)
        print(f'\nModel Mean Average Recalls: {recalls}')
        
    
    ## Take the union of all the predictions row-wise/topic-wise from the models for the same 'k' values
    matched_content_idxs_list = [np_rowise_unique(np.hstack([model_pred[i] for model_pred in model_preds]), fill_value = FILL_VALUE) for i,_ in enumerate(k_vals)]
    return matched_content_idxs_list

# https://stackoverflow.com/questions/26958233/numpy-row-wise-unique-elements
# Below function keeps only the unique numbers in each row and plugs 'fill_value' in places where the duplicates are there
def np_rowise_unique(a, fill_value):
    unique = np.sort(a)
    duplicates = unique[:,  1:] == unique[:, :-1]
    unique[:, 1:][duplicates] = fill_value
    return unique

def get_np_from_retreival_df(df, fill_value):
    l_l = list(df['c_id'].to_numpy())
    max_length = max(map(len, l_l)) # find the maximum list size
    if max_length == 0:
        new_np = np.expand_dims(np.array([fill_value] * len(l_l)), axis = 1)
    else:
        new_np = np.array([xi+[fill_value]*(max_length-len(xi)) for xi in l_l]) # Add -2 into rows that have less than max_length
    return new_np


def get_avg_recall_precision_f2_from_np(correct_np, pred_np):
    # Per row true-positive count
    per_row_TP_count = np_rowise_intersection(correct_np, pred_np).sum(axis=1)

    per_row_TP_count = per_row_TP_count.astype(float)

    # Count TP + FN per row/topic
    per_row_TP_FN_count = (np.sum(np.array(correct_np) != '-1', axis=1)).astype(float) # we have plugged in -ve fill-value in the 2D np array to make sure the row sizes match up. So we ignore those to get a count of TP+FN.
    
    # Count TP + FP per row/topic
    per_row_TP_FP_count = (np.sum(np.array(pred_np) != '-2', axis=1)).astype(float) # we have plugged in -ve fill-value in the 2D np array to make sure the row sizes match up. So we ignore those to get a count of TP+FP.
    

    # Calculate recall per row/topic
    recall = per_row_TP_count/per_row_TP_FN_count

    # Calculate precision per row/topic and sum it up
    # precision = per_row_TP_count/per_row_TP_FP_count # This doesn't work since the denominator can be zeros
    precision = np.divide(per_row_TP_count, per_row_TP_FP_count, out=np.ones_like(per_row_TP_count), where=per_row_TP_FP_count!=0)
    
    # F2 score per row
    numa = (5 * precision * recall)
    deno = (4*precision + recall)
    f2 = np.divide(numa, deno, out=np.zeros_like(numa), where=deno!=0)
    
    n = correct_np.shape[0]

    avg_recall    = round(recall.sum()/n, 3)
    avg_precision = round(precision.sum()/n, 3)
    avg_f2        = round(f2.sum()/n, 3)
    return avg_recall, avg_precision, avg_f2

def compute_retreival_f2_score_from_bin_data(retreival_bin_data_df):
    correct_df = pd.DataFrame(retreival_bin_data_df.loc[retreival_bin_data_df['label'] == 1.0].drop(['label', 'pred'],axis=1).groupby('t_id', group_keys=False)['c_id'].apply(list)).reset_index()
    pred_df    = pd.DataFrame(retreival_bin_data_df.loc[retreival_bin_data_df['pred']  == 1.0].drop(['label', 'pred'],axis=1).groupby('t_id', group_keys=False)['c_id'].apply(list)).reset_index()

    # Make sure ordering of t_id predictions is same as the correct data
    pred_df = pd.merge(correct_df[["t_id"]], pred_df, how="left", on="t_id")
    # Any t_id that doesn't have mapping is filled with empty list
    pred_df['c_id'] = [ [] if x is np.NaN else x for x in pred_df['c_id'] ]

    correct_np = get_np_from_retreival_df(correct_df, fill_value = '-1')
    pred_np    = get_np_from_retreival_df(pred_df, fill_value = '-2')
    return get_avg_recall_precision_f2_from_np(correct_np, pred_np)


def compute_val_scores_for_binary_classification(t_model, device, data_loader, threshold, debug = 0):
    t_model.eval()
    val_acc = 0
    f2_score = 0
    f1_score = 0
    print_done = False
    retreival_bin_data_np = []
    with torch.no_grad():
        for t_id, c_id, features, label in tqdm(data_loader, desc="Validation"):
            features = features.to(device)
            label    = label.to(device)

            logits = t_model(features)
            sig = (torch.sigmoid(logits))
            
            pred = torch.round(sig)
            
            # label = (label == 1.0) # convert label to boolean
            t_id = np.expand_dims(t_id, axis=1)
            c_id = np.expand_dims(c_id, axis=1)
            label_np = np.expand_dims(label.cpu().numpy(), axis=1)
            pred_np = pred.cpu().numpy()
            if (debug!= 0 and not print_done):
                print(logits.size())
                print(logits[0:debug])
                print(sig.size())
                print(sig[0:debug])
                print(pred.size())
                print(pred[0:debug])
                print(label.size())
                print(label[0:debug])
                print(t_id.shape)
                print(c_id.shape)
                print(label_np.shape)
                print(pred_np.shape)
                print_done = True
            # acc  = binary_acc(pred,label, add_sigmoid = False)
            # val_acc  += acc.item()
            # f1, f2 = calc_f2_score(pred.cpu(), label.cpu())
            # f2_score += f2
            # f1_score += f1
            batch_data_np = np.concatenate( [t_id, c_id, label_np, pred_np], axis = 1)
            retreival_bin_data_np.append(batch_data_np)
    retreival_bin_data_np = np.concatenate(retreival_bin_data_np, axis = 0)
    # print(retreival_bin_data_np.shape)
    # Calculate binary accuracy, f1, f2 scores
    retreival_bin_data_df = pd.DataFrame(retreival_bin_data_np, columns = ['t_id', 'c_id', 'label', 'pred']).astype({'t_id': str, 'c_id':str, 'label':np.float32, 'pred':np.float32})
    bin_acc = binary_acc(torch.tensor(retreival_bin_data_df['pred']), torch.tensor(retreival_bin_data_df['label']), add_sigmoid = False).item()
    bin_f1, bin_f2 = calc_f2_score(torch.tensor(retreival_bin_data_df['pred']), torch.tensor(retreival_bin_data_df['label']))
    avg_recall, avg_precision, avg_f2 = compute_retreival_f2_score_from_bin_data(retreival_bin_data_df)
    t_model.train()
    # n = len(data_loader)
    return bin_acc, round(bin_f1, 3), round(bin_f2, 3), avg_recall, avg_precision, avg_f2

#
# Kept below function only for reference. It isn't very useful since retreival val
# don't really reveal much with binary classification data.
#
# def compute_val_scores_for_binary_classification_old_bkp(t_model, device, data_loader, threshold, debug = 0):
#     t_model.eval()
#     val_acc = 0
#     f2_score = 0
#     f1_score = 0
#     print_done = False
#     with torch.no_grad():
#         for features, label in tqdm(data_loader, desc="Validation"):
#             features = features.to(device)
#             label    = label.to(device)

#             logits = t_model(features)
#             sig = (torch.sigmoid(logits))
            
#             pred = torch.round(sig)
            
#             # label = (label == 1.0) # convert label to boolean
            
#             if (debug!= 0 and not print_done):
#                 print(logits.size())
#                 print(logits[0:debug])
#                 print(sig.size())
#                 print(sig[0:debug])
#                 print(pred.size())
#                 print(pred[0:debug])
#                 print(label.size())
#                 print(label[0:debug])
#                 print_done = True
#             acc  = binary_acc(pred,label, add_sigmoid = False)
#             val_acc  += acc.item()
#             f1, f2 = calc_f2_score(pred.cpu(), label.cpu())
#             f2_score += f2
#             f1_score += f1
#     t_model.train()
#     n = len(data_loader)
#     return round(val_acc/n, 2), round(f1_score/n, 3), round(f2_score/n, 3)
