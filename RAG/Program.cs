using LangChain.Databases.Sqlite;
using LangChain.DocumentLoaders;
using LangChain.Providers.Ollama;
using LangChain.Extensions;
using Ollama;
using LangChain.Splitters.Text;
using LangChain.Databases;

// Initialize the provider
var provider = new OllamaProvider(options: new RequestOptions
{
	Stop = new[] { "\n" },
	Temperature = 0.2f
});

// Initialize the embedding model and LLM
var embeddingModel = new OllamaEmbeddingModel(provider, id: "all-minilm");
//var embeddingModel = new OllamaEmbeddingModel(provider, id: "nomic-embed-text");
//var embeddingModel = new OllamaEmbeddingModel(provider, id: "mxbai-embed-large");
//var embeddingModel = new OllamaEmbeddingModel(provider, id: "snowflake-arctic-embed");
var llm = new OllamaChatModel(provider, id: "llama3");

// Initialize the vector database
var vectorDatabase = new SqLiteVectorDatabase(dataSource: "vectors.db");

// Define the text splitter
ITextSplitter recursiveTextSplitter = new RecursiveCharacterTextSplitter(
	separators: new List<string> { "\n\n", "\n", " " },
	chunkSize: 100, 
	chunkOverlap: 10
);

// Function to clear the collection
async Task ClearCollectionAsync(string collectionName)
{
	await vectorDatabase.DeleteCollectionAsync(collectionName);
}

// Function to add documents from a new data source
async Task<IVectorCollection> AddDocumentsFromDataSourceAsync(string dataSourcePath, string collectionName)
{
	await ClearCollectionAsync(collectionName);

	var vectorCollection = await vectorDatabase.AddDocumentsFromAsync<PdfPigPdfLoader>(
		embeddingModel,
		//dimensions: 1536 //Should be 1536 for TextEmbeddingV3SmallModel
		dimensions: 384, //for all-MiniLM- 384 dimensions
		dataSource: DataSource.FromUrl(dataSourcePath),
		collectionName: collectionName,
		textSplitter: recursiveTextSplitter, //null, 
		behavior: AddDocumentsToDatabaseBehavior.OverwriteExistingCollection
	);

	return vectorCollection;
}

// Example of changing the data source
var vectorCollection = await AddDocumentsFromDataSourceAsync("https://canonburyprimaryschool.co.uk/wp-content/uploads/2016/01/Joanne-K.-Rowling-Harry-Potter-Book-1-Harry-Potter-and-the-Philosophers-Stone-EnglishOnlineClub.com_.pdf", "Aircraft");

// Set up a loop for continuous interaction
while (true)
{
	Console.Write("Ask me anything or type 'exit' to stop: ");
	string userQuestion = Console.ReadLine();

	if (userQuestion.ToLower() == "exit")
	{
		break;
	}

	var similarDocuments = await vectorCollection.GetSimilarDocuments(embeddingModel, userQuestion, amount: 3);

	var response = await llm.GenerateAsync(
		$"""
        Use the following pieces of context to answer the question at the end.
        If the answer is not in context then use your own knowledge to provide the best possible answer.
        Keep the answer as short as possible.

        {similarDocuments.AsString()}

        Question: {userQuestion}
        Helpful Answer:
        """).ConfigureAwait(false);

	Console.WriteLine($"LLM answer: {response}");

	// Optionally, write out the vector db similar documents if needed for debugging
	Console.WriteLine("Similar Documents:");
	foreach (var document in similarDocuments)
	{
		Console.WriteLine(document);
	}
}