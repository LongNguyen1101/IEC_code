def get_config(data_path, directory_encode_category='encoding_category_data', 
               impute_data='impute_data', directory_scale='scale_data' , 
               metadata='metadata.txt', drop_dup=True, smote_seed=11):
    return  {
        'data_path': data_path,
        'directory_encode_category': directory_encode_category,
        'directory_impute': impute_data,
        'directory_scale': directory_scale,
        'metadata': metadata,
        'smote_seed': smote_seed,
        'drop_dup': drop_dup
    }