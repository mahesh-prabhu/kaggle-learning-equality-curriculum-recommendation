import pandas as pd
from datasets import Dataset
from l_utils import *

def retreiver_topics_FE(topics_df, sep_token, col_sep_token):
    # Topic feature engineering

    text_emb_cols = ['title', 'description']
    # Remove NaNs from title, description
    for col in text_emb_cols:
        topics_df[col].fillna('',inplace=True)


    topics_parent_dict = pd.Series(topics_df.parent.values,index=topics_df.id).to_dict()
    topics_title_dict = pd.Series(topics_df.title.values,index=topics_df.id).to_dict()

    topics_df['parents_title'] = topics_df.apply(lambda r: get_all_parents_title_string(topics_parent_dict, topics_title_dict, r['id'], sep_token), axis = 1)

    # Drop rows that have has_content == False
    topics_df = (topics_df.loc[topics_df["has_content"] == True]).copy().reset_index(drop = True)

    topics_df['combined'] = "topic" + col_sep_token + topics_df['title'] + col_sep_token + topics_df['description']  + col_sep_token + topics_df['parents_title'] + col_sep_token

    # Drop unused columns
    text_emb_cols = ['title', 'description'] + ['parents_title']
    # text_emb_cols = ['title', 'parents_title', 'description']
    remove_cols = list_diff_with_ordering_maintained(topics_df.columns, ['id', 'combined'])
    topics_df = topics_df.drop(remove_cols, axis=1)

    return topics_df

def retreiver_contents_FE(contents_df, sep_token, col_sep_token):
    # Content feature engineering

    # Get text embeddings of title, description and text
    text_emb_cols = ['title', 'description', 'text']
    # Remove NaNs from title, description and text
    for col in text_emb_cols:
        contents_df[col].fillna('',inplace=True)

    contents_df['combined'] = "content" + col_sep_token + contents_df['title'] + col_sep_token + contents_df['description'] + col_sep_token + contents_df['text'] + col_sep_token

    # Drop unused columns
    remove_cols = list_diff_with_ordering_maintained(contents_df.columns, ['id', 'combined'])
    contents_df = contents_df.drop(remove_cols, axis=1)

    return contents_df

def ranker_topics_FE(topics_df, sep_token, col_sep_token):
    # Topic feature engineering

    text_emb_cols = ['title', 'description']
    # Remove NaNs from title, description
    for col in text_emb_cols:
        topics_df[col].fillna('',inplace=True)


    topics_parent_dict = pd.Series(topics_df.parent.values,index=topics_df.id).to_dict()
    topics_title_dict = pd.Series(topics_df.title.values,index=topics_df.id).to_dict()

    topics_df['parents_title'] = topics_df.apply(lambda r: get_all_parents_title_string(topics_parent_dict, topics_title_dict, r['id'], sep_token), axis = 1)

    # Drop rows that have has_content == False
    topics_df = (topics_df.loc[topics_df["has_content"] == True]).copy().reset_index(drop = True)

    topics_df['combined'] = "topic" + col_sep_token + topics_df['title'] + col_sep_token + topics_df['description']  + col_sep_token + topics_df['parents_title']

    # Drop unused columns
    text_emb_cols = ['title', 'description'] + ['parents_title']
    # text_emb_cols = ['title', 'parents_title', 'description']
    remove_cols = list_diff_with_ordering_maintained(topics_df.columns, ['id', 'combined'])
    topics_df = topics_df.drop(remove_cols, axis=1)

    return topics_df

def ranker_contents_FE(contents_df, sep_token, col_sep_token):
    # Content feature engineering

    # Get text embeddings of title, description and text
    text_emb_cols = ['title', 'description', 'text']
    # Remove NaNs from title, description and text
    for col in text_emb_cols:
        contents_df[col].fillna('',inplace=True)

    contents_df['combined'] = "content" + col_sep_token + contents_df['title'] + col_sep_token + contents_df['description'] + col_sep_token + contents_df['text']

    # Drop unused columns
    remove_cols = list_diff_with_ordering_maintained(contents_df.columns, ['id', 'combined'])
    contents_df = contents_df.drop(remove_cols, axis=1)

    return contents_df

def tokenize_batch(batch, tokenizer, text_cols, col_prefix, max_length):
    """
    Tokenizes the dataset on the specific columns, truncated/padded to a max length.
    Adds the suffix "col_prefix{i}" to the input ids and attention mask of the content texts.
    
    """
    sep = tokenizer.sep_token


    text_lists   = [batch[c] for c in text_cols]

    tokenized_texts = [
        tokenizer(texts, truncation=True, max_length=max_length, padding='max_length')
        for texts in text_lists
    ] # list of dicts

    new_tokenized_texts = {} # dict
    count = 0

    for tokenized_text_dict in tokenized_texts:
        # Remove token_type_ids. They will just cause errors.
        if 'token_type_ids' in tokenized_text_dict:
            del tokenized_text_dict["token_type_ids"]
        # Uniqufy dict keys and add them to a common dict
        for k, v in tokenized_text_dict.items():
            new_tokenized_texts[f'{col_prefix}{count}_{k}'] = v
            count += 1

    return new_tokenized_texts
    
def get_tokenized_ds(ds, tokenizer, text_cols, col_prefix, max_length=64, debug=False):

    if debug:
        ds = ds.shuffle().select(range(5000))

    tokenized_ds = ds.map(
        tokenize_batch,
        batched=True,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            # topic_cols=[f"topic_{c}" for c in CFG.topic_cols],
            # content_cols=[f"content_{c}" for c in CFG.content_cols],
            text_cols = text_cols,
            col_prefix = col_prefix, 
            max_length=max_length,
        ),
        remove_columns=(list_diff_with_ordering_maintained(ds.column_names, ['id', 'language'])),
        # num_proc=CFG.num_proc,
        num_proc = 2
    )

    return tokenized_ds

def fe_and_tokenize_df(tokenizer, feature_eng_fn, post_process_df, text_cols, col_prefix, max_length, debug, df):

    sep_token     = tokenizer.sep_token
    col_sep_token = tokenizer.sep_token + tokenizer.cls_token
    
    df = feature_eng_fn(df, sep_token, col_sep_token)
    ds = Dataset.from_pandas(df)
    tokenized_ds = get_tokenized_ds(ds, tokenizer, text_cols = text_cols, col_prefix = col_prefix, max_length=max_length, debug=debug)
    tokenized_df = Dataset.to_pandas(tokenized_ds)
    if (post_process_df is not None):
        tokenized_df = post_process_df(tokenized_df)
    return tokenized_df
    

def tokenize_df_and_write_to_parquet(tokenizer, feature_eng_fn, df, text_cols, col_prefix, max_length, tokenized_ds_name, debug = False, post_process_df = None):

    tokenized_df = fe_and_tokenize_df(tokenizer, feature_eng_fn, post_process_df, text_cols, col_prefix, max_length, debug, df)
    tokenized_df.to_parquet(tokenized_ds_name)

#
# I am assuming sep_input_ids is a combo of sep_token followed by cls_token
#
def find_col_sep_indices_in_input_ids(input_ids, sep_input_ids, num_seperators):
    indices = []
    input_ids_len = len(input_ids)
    for cur_idx, input_id in enumerate(input_ids):
        if input_id == sep_input_ids[0]:
            if (cur_idx+1 < input_ids_len and (input_ids[cur_idx+1] == sep_input_ids[1])):
                indices.append(cur_idx+1)
        if (cur_idx+1 == input_ids_len and (len(indices) < num_seperators)):
            indices.append(cur_idx+1)
    indices_len = len(indices)
    if indices_len < num_seperators:
        indices += [indices[indices_len-1]]*(num_seperators-indices_len)
    return indices

def find_col_sep_indices(df, sep_input_ids, num_seperators):
    input_id_cols = [col for col in df.columns if col.endswith("input_ids")]
    for col in input_id_cols:
        df[col + '_sep_indices'] = df[col].apply(lambda input_ids: find_col_sep_indices_in_input_ids(input_ids, sep_input_ids, num_seperators))
    return df


def add_col_sep_ids_and_pad(input_ids, start_tok_id, end_tok_id, pad_tok_id, num_features, new_max_len):
    """
    Function adds extra sep_token_ids if necessary, and force padding to new_max_len =  orig_max_len + len(col_sep_token_ids)*num_col_sep_toks
    """
    feature_count = 0
    new_input_ids = []
    input_ids_len = len(input_ids)
    for cur_idx, input_id in enumerate(input_ids):
        if input_id == pad_tok_id:
            break
        if input_id == end_tok_id:
            if (cur_idx+1 < input_ids_len):
                if (input_ids[cur_idx+1] == start_tok_id) or (input_ids[cur_idx+1] == pad_tok_id):
                    feature_count += 1
            else:
                feature_count += 1
        new_input_ids.append(input_id)
    if feature_count < num_features:
        # Add "empty strings" before any padding tokens
        new_input_ids += [start_tok_id, end_tok_id]*(num_features-feature_count)

    if (len(new_input_ids) < new_max_len):
        new_input_ids += [pad_tok_id]*(new_max_len-len(new_input_ids))
    return new_input_ids

def update_attention_mask(input_ids, pad_tok_id):
    """
    Add extra padding token to match new_max_len
    """
    return [0 if tok == pad_tok_id else 1 for tok in input_ids]

def pad_extra_sep_toks(df, col_prefix, start_tok_id, end_tok_id, pad_tok_id, num_features, new_max_len):
    """
    We need to make sure there are num_col_sep_toks number of tokens in each row.
    This way if one of column features is empty then we will explicitly have a representation
    for it. We extend max_len to orig_max_len + len(col_sep_token_ids)*num_col_sep_toks, 
    by adding pad tokens at the end.
    *** I am assuming len(col_sep_token_ids) is 2 ***
    """

    input_id_col = f'{col_prefix}0_input_ids'
    attn_mask_col = f'{col_prefix}1_attention_mask'
    df[input_id_col] = df[input_id_col].apply(lambda input_ids: add_col_sep_ids_and_pad(input_ids, start_tok_id, end_tok_id, pad_tok_id, num_features, new_max_len))
    df[attn_mask_col] = df[input_id_col].apply(lambda input_ids: update_attention_mask(input_ids, pad_tok_id))

def ranker_tokenize_main(tokenizer, topics_df, contents_df, max_length, debug = False, show_progress_bars=True):
    
    progress_bar_control(show_progress_bars)
    
    NUM_FEATURES = 4

    new_max_len = max_length + 2*(NUM_FEATURES-2) # I am assuming we'll be looking for start_tok_id(cls_token) ... end_tok_id(sep_token) for each feature. The tokenized string will have at least 2 features at minimum.
    
    topics_tok_df = fe_and_tokenize_df(tokenizer, ranker_topics_FE, post_process_df = None, text_cols = ['combined'], col_prefix = "t_", max_length = max_length, debug = False, df = topics_df)

    # print(topics_tok_df.columns)

    pad_extra_sep_toks(topics_tok_df, "t_", tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, NUM_FEATURES, new_max_len)

    contents_tok_df = fe_and_tokenize_df(tokenizer, ranker_contents_FE, post_process_df = None, text_cols = ['combined'], col_prefix = "c_", max_length = max_length, debug = False, df = contents_df)

    # print(contents_tok_df.columns)
    
    pad_extra_sep_toks(contents_tok_df, "c_", tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, NUM_FEATURES, new_max_len)
    return topics_tok_df, contents_tok_df


# Merge the train and val corr data into tokenized topic, content data
def combine_input_ids(toks, pad_tok_id):
    """
    Since we concatenate the topic and content input_ids, we need to make sure
    the padding appears only at the end after the concatenation
    """
    # toks is of numpy array type, not list
    # toks = toks.tolist()
    toks_len = len(toks)
    for i, e in enumerate(toks):
        if e == pad_tok_id:
            break
        elif i >= toks_len//2:
            break
    if i < toks_len//2:
        new_toks = toks[:i] + toks[toks_len//2:]
    else:
        new_toks = toks
    #new_toks = np.concatenate(new_toks, np.repeat(pad_tok_id, (toks_len - len(new_toks))))
    new_toks += [pad_tok_id] * (toks_len - len(new_toks))
    return new_toks

# def update_attention_mask(input_ids, pad_tok_id):
#     """
#     Add extra padding token to match new_max_len
#     """
#     return [0 if tok == pad_tok_id else 1 for tok in input_ids]

def get_ranker_topic_content_merged_data(tokenizer, corr_ids_df, topics_tokenized_df, contents_tokenized_df, keep_ids = False):

    pt('Merging data ... ')
    merged_df = pd.merge(corr_ids_df, topics_tokenized_df, how = 'inner', left_on = 't_id', right_on = 'id').drop(['id'], axis=1)
    if not keep_ids:
        merged_df = merged_df.drop(['t_id'], axis=1)
    merged_df = pd.merge(merged_df, contents_tokenized_df, how = 'inner', left_on = 'c_id', right_on = 'id').drop(['id'], axis = 1)
    if not keep_ids:
        merged_df = merged_df.drop(['c_id'], axis=1)

    pt('Concatenating input ids ... ')
    merged_df['input_ids'] = merged_df['t_0_input_ids'].apply(lambda r: r if type(r) == list else r.tolist()) +  merged_df['c_0_input_ids'].apply(lambda r: r if type(r) == list else r.tolist())

    pt('Adjusting padding for concatenated input ids ... ')
    merged_df['input_ids'] = merged_df['input_ids'].apply(lambda row: combine_input_ids(row, tokenizer.pad_token_id))

    pt('Creating new attention mask ... ')
    merged_df['attention_mask'] = merged_df['input_ids'].apply(lambda r: update_attention_mask(r, tokenizer.pad_token_id))
    
    merged_df = merged_df.drop(['t_0_input_ids', 't_1_attention_mask', 'c_0_input_ids', 'c_1_attention_mask'],axis = 1)

    return merged_df
