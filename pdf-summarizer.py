"""
This script is designed to summarize PDF documents using a Vision-Language Model (VLM).

/home/nikunj/Documents/stellarium_user_guide.pdf

"""

# import necessary libraries
import argparse
import sys
import torch
import traceback
from byaldi import RAGMultiModalModel
from pdf2image import convert_from_path
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_info()

def main(args):

    # convert uploaded PDF to images list
    try:
        pdf_path = args["pdf_path"]
        images = convert_from_path(pdf_path)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        traceback.print_exc()
        return

    # Load the pretranied ColPali model
    try:
        rag = RAGMultiModalModel.from_pretrained("vidore/colpali")
    except Exception as e:
        print(f"Error loading the ColPali model: {e}")
        traceback.print_exc()
        return

    # Index the loaded PDF document using the RAG ColPali model
    # input_path: this parameter takes in the PDF file path
    # index_name: this parameter is used to name the index created for the PDF document which is later used for querying
    # store_collection_with_index: if set to False, the document will not be stored along with the index reducing the storage space
    # overwrite: if set to True, it will overwrite the existing index with the same name if it exists
    # Below step helps for quick and accurate retrieval of information from the PDF document which is indexed by ColPali model
    try:
        rag.index(input_path=pdf_path,
                index_name="image_index",
                store_collection_with_index=False,
                overwrite=True)
        print(f"PDF document indexed successfully.\n {rag}")
    except Exception as e:
        print(f"Error indexing the PDF document: {e}")
        traceback.print_exc()
        return

    # Query the document based on a question
    # the rag.search method is used to search the indexed document using the text query and we tell the model to return only top k results
    # k: this parameter is used to specify the number of top results to be returned
    text_query = "What is the Sun in our solar system?"
    try:
        results = rag.search(text_query, k=10)
        print(f"Query results for the search: '{results}':")
        print(f"The retrieved image results are: {images[results[0]["page_num"] - 1]}")
    except Exception as e:
        print(f"Error querying the indexed document: {e}")
        traceback.print_exc()
        return

    # Load Llava Gemma 2b model. This will be used to input multimodal data (text and images) and generate a text responses.
    try:
        # checkpoint = "Intel/llava-gemma-2b"   # too large for 8GB VRAM GPU
        checkpoint = "Salesforce/blip2-opt-2.7b"  # "Salesforce/blip2-flan-t5-xl"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Blip2ForConditionalGeneration.from_pretrained(checkpoint).to(device)   # load the Llava Gemma 2b model from its checkpoint
        processor = AutoProcessor.from_pretrained(checkpoint)   # the processor preprocesses the input data before passing it to the model
        print(f"Llava Gemma 2b model loaded successfully.")
    except Exception as e:
        print(f"Error loading the Llava Gemma 2b model: {e}")
        traceback.print_exc()
        return

    # Prepare the input for the Llava model
    # apply_chat_template is used to structure the input for the Llava model in a chat like conversation format using the template.
    # tokenize mentions that the input includes and image followed by the users text query
    # add_generation_prompt: if set to True, it signals the model to generate a response based on the input
    try:
        # prompt = processor.tokenizer.apply_chat_template(
        #     [{"role": "user", "content": f"<image>\n{text_query}"}],
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        prompt = text_query
        image_index = results[0]["page_num"] - 1  # get the index of the image corresponding to the query result
        image = images[image_index]  # retrieve the image from the list of images
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)  # prepare the input for the Llava model
    except Exception as e:
        print(f"Error preparing the input for the Llava model: {e}")
        traceback.print_exc()
        return

    # Generate the response using the Llava model
    try:
        generate_ids = model.generate(**inputs, max_length=512)
        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(f"The generated response is: {output}")
    except Exception as e:
        print(f"Error generating the response: {e}")
        traceback.print_exc()
        return

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Summarize PDF documents using a Vision-Language Model.")
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file to be summarized.")
    args = vars(parser.parse_args())

    main(args)

    sys.exit(0)


