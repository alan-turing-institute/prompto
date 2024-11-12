# import argparse
# import json
# import os

# from prompto.rephrasal import Rephraser, load_rephrase_folder
# from prompto.utils import parse_list_arg


# def obtain_output_filepath(input_filepath: str, output_folder: str) -> str:
#     input_filename = os.path.basename(input_filepath)
#     out_filepath = os.path.join(output_folder, f"rephrasal-{input_filename}")
#     return out_filepath


# def main():
#     """
#     Generate and run a file for the expanding the prompts in an experiment
#     file using a model to rephrase the prompts.
#     """
#     # parse command line arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--input-file",
#         "-i",
#         help="Path to the input experiment file",
#         type=str,
#         required=True,
#     )
#     parser.add_argument(
#         "--rephrase-folder",
#         "-l",
#         help=(
#             "Location of the rephrase folder storing the template.txt "
#             "and settings.json to be used"
#         ),
#         type=str,
#         required=True,
#     )
#     parser.add_argument(
#         "--templates",
#         "-t",
#         help=(
#             "Template file to be used for the rephrasals. "
#             "This must be .txt files in the rephrase folder. "
#             "By default, the template file is 'template.txt'"
#         ),
#         type=str,
#         default="template.txt",
#     )
#     parser.add_argument(
#         "--rephrase-model",
#         "-r",
#         help=(
#             "Rephrase models(s) to be used separated by commas. "
#             "These must be keys in the rephrase settings dictionary"
#         ),
#         type=str,
#         required=True,
#     )
#     parser.add_argument(
#         "--output-folder",
#         "-o",
#         help="Location where the rephrasal file will be created",
#         type=str,
#         default="./",
#     )
#     args = parser.parse_args()

#     # parse input file
#     input_filepath = args.input_file
#     try:
#         with open(input_filepath, "r") as f:
#             responses = [dict(json.loads(line)) for line in f]
#     except FileNotFoundError as exc:
#         raise FileNotFoundError(
#             f"Input file '{input_filepath}' is not a valid input file"
#         ) from exc

#     # parse template, rephrase folder and rephrase arguments
#     template_prompts, rephrase_settings = load_rephrase_folder(
#         rephrase_folder=args.rephrase_folder, templates=args.templates
#     )
#     rephrase_model = parse_list_arg(argument=args.rephrase_model)
#     # check if the rephrase model is in the rephrase settings dictionary
#     Rephraser.check_rephrase_model_in_rephrase_settings(
#         rephrase_model=rephrase_model, rephrase_settings=rephrase_settings,
#     )

#     # create output file path name
#     out_filepath = obtain_output_filepath(
#         input_filepath=input_filepath, output_folder=args.output_folder
#     )

#     # create rephraser object from the parsed arguments
#     r = Rephraser(
#         completed_responses=responses,
#         template_prompts=template_prompts,
#         rephrase_settings=rephrase_settings,
#     )

#     # create rephrase file
#     r.create_rephrase_file(rephrase_model=rephrase_model, out_filepath=out_filepath)


# if __name__ == "__main__":
#     main()
