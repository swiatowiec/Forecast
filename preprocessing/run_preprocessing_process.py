import argparse
from application.facade import PreprocessingFitTransformFacade, PreprocessingFitTransformArgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir_path', type=str, default='data/input')
    parser.add_argument('--output_dir_path', type=str,
                        default='data/preprocessed')
    parser.add_argument('--metadata_dir_path', type=str,
                        default='preprocessing/artifacts/metadata')

    arg = parser.parse_args()
    args = PreprocessingFitTransformArgs(input_dir_path=arg.input_dir_path,
                                         output_dir_path=arg.output_dir_path)


    preprocess = PreprocessingFitTransformFacade()

    preprocess.fit_transform(args)