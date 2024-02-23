import gradio as gr
from advanced_rag import ElevatedRagChain


rag_chain = ElevatedRagChain()


def load_pdfs(pdf_links):
    if not pdf_links:
        gr.Warning("Please enter non-empty URLs")
        return "Please enter non-empty URLs"
    try:
        pdf_links = pdf_links.split("\n")  # get individual PDF links
        rag_chain.add_pdfs_to_vectore_store(pdf_links)
        gr.Info("PDFs loaded successfully into a new vector store. If you had an old one, it was overwritten.")
        return "PDFs loaded successfully into a new vector store. If you had an old one, it was overwritten."
    except Exception as e:
        gr.Warning("Could not load PDFs. Are URLs valid?")
        print(e)
        return "Could not load PDFs. Are URLs valid?"


def submit_query(query):
    if not query:
        gr.Warning("Please enter a non-empty query")
        return "Please enter a non-empty query"
    if hasattr(rag_chain, 'elevated_rag_chain'):
        try:
            response = rag_chain.elevated_rag_chain.invoke(query)
            return response
        except Exception as e:
            gr.Warning("LLM error. Please re-submit your query")
            print(e)
            return "LLM error. Please re-submit your query"

    else:
        gr.Warning("Please load PDFs before submitting a query")
        return "Please load PDFs before submitting a query"


def reset_app():
    global rag_chain
    rag_chain = ElevatedRagChain()  # Re-initialize the ElevatedRagChain object
    gr.Info("App reset successfully. You can now load new PDFs")
    return "App reset successfully. You can now load new PDFs"


# custom css for different age elements
custom_css = """
// customize button
button {
    background-color: grey !important;
    font-family: Arial !important;
    font-weight: bold !important;
    color: blue !important;
}



// customize background color and use it as "app = gr.Blocks(css=custom_css)"
//.gradio-container {background-color: #E0F7FA}
"""

# Define the Gradio app using Blocks for a flexible layout
app = gr.Blocks(css=custom_css)    # theme=gr.themes.Base(), Soft(), Default(), Glass(), Monochrome(): https://www.gradio.app/guides/theming-guide

with app:
    gr.Markdown('''# Query your own data
## Llama 2 RAG
- Type in one or more URLs for PDF files - one per line and click on Load PDFs. Wait until the RAG system is built.
- Type your query and click on Submit Query. Once the LLM sends back a reponse, it will be displayed in the Reponse field.
- The system "remembers" the source documents, but has no memory of past user queries.
- Click on Reset App to clear / reset the RAG system
    ''')
    with gr.Row():
        with gr.Column():
            pdf_input = gr.Textbox(label="Enter your PDF URLs (one per line)", placeholder="Enter one URL per line", lines=4)
            load_button = gr.Button("Load PDF")
        with gr.Column():
            query_input = gr.Textbox(label="Enter your query here", placeholder="Type your query", lines=4)
            submit_button = gr.Button("Submit")
            
    response_output = gr.Textbox(label="Response", placeholder="Response will appear here", lines=4)
    reset_button = gr.Button("Reset App")

    load_button.click(load_pdfs, inputs=pdf_input, outputs=response_output)
    submit_button.click(submit_query, inputs=query_input, outputs=response_output)
    reset_button.click(reset_app, inputs=None, outputs=response_output)


# Run the app
app.launch()
