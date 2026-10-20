import json
import ollama

from risk_analyser import ConstructionRiskAnalyser
import os
from dotenv import load_dotenv

load_dotenv()


def main():

    # -----------------------------------------
    # Load configuration
    # -----------------------------------------

    csv_file_path = os.getenv("CSV_FILE_PATH")
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
    chroma_db_path = os.getenv("VECTOR_DB_PATH")
    chroma_db_collection_name = os.getenv("VECTOR_COLLECTION_NAME")
    query_collection_name = os.getenv("QUERY_VECTOR_COLLECTION_NAME")

    # -----------------------------------------
    # Create analyzer
    # -----------------------------------------

    analyser = ConstructionRiskAnalyser(
        embedding_model_name,
        chroma_db_path,
        chroma_db_collection_name,
        query_collection_name,
    )

    # knowledge_embed = analyser.save_text_to_chroma_db()

    # -----------------------------------------
    # Start conversation
    # -----------------------------------------

    print("\n==========================================")
    print("   Construction Risk Analyzer")
    print("==========================================")
    print("Ask questions about construction project risks.")
    print("Type 'exit' or 'quit' to stop.")
    print("==========================================\n")

    conversation_history = []
    # user_query = "Give me the list of project names in North region"

    while True:
        # -----------------------------------------
        # Get user question
        # -----------------------------------------

        user_query = input("You: ").strip()

        # results = analyser.query_query_knowledge(user_query, top_k=5)
        # query_text_result = analyser.prepare_query_knowledge_context(results)
        # query_plan = analyser.generate_pandas_query(user_query, query_text_result)
        # df = analyser.data_load(csv_file_path)
        # print(query_plan)
        # df = analyser.execute_query_plan(query_plan)
        # print("analyze query: ", df)

        # Ignore empty input
        if not user_query:
            continue

        # Exit
        if user_query.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

        try:
            route = analyser.route_query(user_query)

            # print("\nROUTE:")
            # print(route)

            route_plan = json.loads(route)

            intent = route_plan["intent"]

            print("\nINTENT:")
            print(intent)

            if intent == "STRUCTURED":
                query_knowledge = analyser.query_query_knowledge(user_query, top_k=5)

                query_context = analyser.prepare_query_knowledge_context(
                    query_knowledge
                )

                query_plan = analyser.generate_pandas_query(user_query, query_context)

                # print("\nQUERY PLAN:")
                # print(query_plan)

                df = analyser.data_load(csv_file_path)

                result = analyser.execute_query_plan(query_plan)

                print("\nRESULT:")
                print(result)

            elif intent == "SEMANTIC":
                results = analyser.query_chroma_db(user_query)

                context = analyser.prepare_context(results)

                summary = analyser.summarize_risks(context, user_query)

                print("\nANSWER:")
                print(summary)

            elif intent == "HYBRID":
                print("HYBRID query detected")

                # 1. Apply structured conditions with Pandas.
                # 2. Use the semantic_question for ChromaDB.
                # 3. Intersect/merge the candidate risk IDs.
                # 4. Send the final evidence to the LLM.

                result = analyser.execute_hybrid_query(
                    route_plan=route_plan, csv_file_path=csv_file_path, top_k=10
                )

                # -----------------------------------------------------
                # Display result
                # -----------------------------------------------------

                if result.get("success"):
                    print(
                        "\nStructured candidates :", result.get("structured_count", 0)
                    )

                    print("Semantic candidates   :", result.get("semantic_count", 0))

                    print("Final matched risks   :", result.get("final_count", 0))

                    print("\nFinal Answer:")
                    print(result.get("answer", ""))

                else:
                    print("\nHybrid query failed:")

                    print(result.get("message", "Unknown error"))

            elif intent == "GENERAL":
                response = ollama.chat(
                    model="llama3.2", messages=[{"role": "user", "content": user_query}]
                )

                print("\nANSWER:")
                print(response["message"]["content"])

        except Exception as ex:
            print("\nError:", ex)
            print("Please try again.\n")

    # -----------------------------------------
    # Step 1: Retrieve from ChromaDB
    # -----------------------------------------

    ###results = analyser.query_chroma_db(user_query)

    # -----------------------------------------
    # Step 2: Prepare context
    # -----------------------------------------

    # print("Result length: ", len(results['ids'][0]))
    # print("Project names: ", results['documents'][0])
    ###context = analyser.prepare_context(results)

    # -----------------------------------------
    # Step 3: Generate LLM summary
    # -----------------------------------------

    ###summary = analyser.summarize_risks(context, user_query)

    # -----------------------------------------
    # Step 4: Display answer
    # -----------------------------------------

    ###print("\nRisk Analyzer:\n")
    ###print(summary)
    ###print()

    # -----------------------------------------
    # Step 5: Store conversation
    # -----------------------------------------

    ###conversation_history.append({"user": user_query, "assistant": summary})

    ###except Exception as ex:
    ###print("\nError:", ex)
    ###print("Please try again.\n")


if __name__ == "__main__":
    main()
