using LangChain.Databases.Sqlite;
using LangChain.DocumentLoaders;
using LangChain.Providers.Ollama; 
using LangChain.Extensions;
using Ollama;

var provider = new OllamaProvider(options: new RequestOptions
{
	Stop = new[] { "\n" }, // This option specifies the stop sequence for the model's output. Here, it stops generating text when it encounters a newline character (\n).
	Temperature = 0.0f // Set to 0.0f, indicating that the model should provide deterministic, less varied responses, focusing on the most likely output.
});

var embeddingModel = new OllamaEmbeddingModel(provider, id: "all-minilm");
//var embeddingModel = new OllamaEmbeddingModel(provider, id: "nomic-embed-text");
//var embeddingModel = new OllamaEmbeddingModel(provider, id: "mxbai-embed-large")
var llm = new OllamaChatModel(provider, id: "llama3");
var vectorDatabase = new SqLiteVectorDatabase(dataSource: "vectors.db"); //This initializes a database for storing vectors. These vectors are numerical representations of text data that can be efficiently searched and compared.
																		 //The database is stored in a file named vectors.db 

var vectorCollection = await vectorDatabase.AddDocumentsFromAsync<PdfPigPdfLoader>( // This function call adds documents to the vector database. The documents are loaded from a PDF file using the PdfPigPdfLoader class.
	embeddingModel, // The text extracted from the PDF is converted into embeddings using the specified model (all-minilm), which allows for semantic searching of the text.	
	dimensions: 1536, // Should be 1536 for TextEmbeddingV3SmallModel
	//dimensions: 384, //for all-MiniLM- 384 dimensions
	dataSource: DataSource.FromUrl("https://canonburyprimaryschool.co.uk/wp-content/uploads/2016/01/Joanne-K.-Rowling-Harry-Potter-Book-1-Harry-Potter-and-the-Philosophers-Stone-EnglishOnlineClub.com_.pdf"),
	collectionName: "harrypotter", // Names the collection in the database where these documents are stored, which is useful for maintaining separate datasets within the same database.
	textSplitter: null, // This would typically define how the text should be split into segments for processing, but it's null here, suggesting that the default splitting mechanism is used.
	behavior: AddDocumentsToDatabaseBehavior.JustReturnCollectionIfCollectionIsAlreadyExists);


// Set up a loop for continuous interaction
while (true)
{
	Console.Write("Ask me anything: ");
	string userQuestion = Console.ReadLine();

	if (userQuestion.ToLower() == "exit")
	{
		break;
	}

	var similarDocuments = await vectorCollection.GetSimilarDocuments(embeddingModel, userQuestion, amount: 1);

	var response = await llm.GenerateAsync(
		$"""
        Use the following pieces of context to answer the question at the end.
        If the answer is not in context then just say that you don't know, don't try to make up an answer.
        Keep the answer as short as possible.

        {similarDocuments.AsString()}

        Question: {userQuestion}
        Helpful Answer:
        """).ConfigureAwait(false);

	Console.WriteLine($"LLM answer: {response}"); //This function call converts the documents retrieved from the vector database into a string format.
												  //These documents are assumed to be relevant to the question and serve as context for the model to base its response on.
												  //Question: {userQuestion}: This is where the user's question is placed. By formatting it this way, you're clearly delineating the query for the model.
												  //Helpful Answer: This is where the model's response will be placed. The model will generate a response based on the context provided and the user's query.

	// Optionally, write out the vectordb similar documents if needed for debugging
	//Console.WriteLine("Similar Documents:");
	//foreach (var document in similarDocuments)
	//{
	//	Console.WriteLine(document);
	//}
}

