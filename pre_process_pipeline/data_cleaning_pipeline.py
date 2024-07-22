from encode_pipeline import EncodePipeline
from impute_pipeline import ImputePipeline
from scale_pipeline import ScalePipeline

def DataCleaningPipeline(config):
    try:
        # encode categorical data -> 7 datasets
        data_pipeline = EncodePipeline(config)
        data_pipeline.run()
        del data_pipeline
        print('Encode category columns successful')

        # impute missing data -> 14 datasets
        impute_pipeline = ImputePipeline(config)
        impute_pipeline.run()
        del impute_pipeline
        print('Impute numerical columns successful')

        # scale data
        scale_pipeline = ScalePipeline(config)
        scale_pipeline.run()
        del scale_pipeline
        print('Scale data successful')
    except RuntimeError as e:
        raise RuntimeError(e)
    
    return True