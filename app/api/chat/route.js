import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `
You are an AI assistant designed to help students find and evaluate computer science professors at UC Davis. Your primary function is to provide information about professors based on student queries, using a vector database that contains each professor's name, rating, classes taught, and reviews.

Your responsibilities include:

1. Interpreting student queries about UC Davis computer science professors.
2. Searching the vector database to find relevant information based on the query.
3. Providing accurate and helpful responses using the Retrieval-Augmented Generation (RAG) approach.
4. Offering insights on professors' teaching styles, course difficulty, and overall student satisfaction.
5. Maintaining a neutral and objective tone when discussing professors and their ratings.
6. Respecting student privacy and not sharing any personal information from reviews.

When responding to queries:
- Always base your responses on the information available in the vector database.
- If asked about a professor or course not in the database, politely inform the student that you don't have information on that specific query.
- Provide a summary of the most relevant information, including the professor's name, overall rating, courses taught, and a brief overview of student sentiment from reviews.
- If asked for more details, offer to provide specific review quotes or additional information from the database.
- Encourage students to consider multiple factors when choosing a professor, not just ratings alone.

Remember, your goal is to assist students in making informed decisions about their computer science courses at UC Davis by providing accurate, up-to-date, and relevant information from your vector database.
`;

export async function POST(req) {
    try {
        const data = await req.json();
        
        // Initialize Pinecone and OpenAI clients with environment variables
        const pc = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY,
        });
        
        const index = pc.index('rag2').namespace('ns2');
        const openai = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY,
        });

        const text = data[data.length - 1].content;

        // Generate text embedding
        const embedding = await openai.embeddings.create({
            model: 'text-embedding-3-small',
            input: text,
        });

        // Query Pinecone index
        const results = await index.query({
            topK: 3,
            includeMetadata: true,
            vector: embedding.data[0].embedding,
        });

        let resultString = '\n\nReturned results from vector db (done automatically): ';
        results.matches.forEach((match) => {
            const classesList = match.metadata.classes.join(', ');
    
            resultString += `
            Professor: ${match.id}
            Review: ${match.metadata.review}
            Classes: ${classesList}
            Stars: ${match.metadata.stars}\n\n
            `;
        });

        const lastMessage = data[data.length - 1];
        const lastMessageContent = lastMessage.content + resultString;
        const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

        // Create a chat completion
        const completion = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                { role: "system", content: systemPrompt },
                ...lastDataWithoutLastMessage,
                { role: 'user', content: lastMessageContent },
            ],
            stream: true,
        });

        // Stream the response
        const stream = new ReadableStream({
            async start(controller) {
                const encoder = new TextEncoder();
                try {
                    for await (const chunk of completion) {
                        const content = chunk.choices[0]?.delta?.content;
                        if (content) {
                            controller.enqueue(encoder.encode(content));
                        }
                    }
                } catch (err) {
                    console.error("Error while streaming response:", err); // Log streaming error
                    controller.error(err);
                } finally {
                    controller.close();
                }
            },
        });

        return new NextResponse(stream);
    } catch (error) {
        console.error("Error occurred during the request:", error); // Log the error details
        return new NextResponse(`Internal Server Error: ${error.message}`, { status: 500 });
    }
}
