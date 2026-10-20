from ast import operator

import pandas as pd
import numpy as np
import ollama
import chromadb
import json
import re


class ConstructionRiskAnalyser:
    def __init__(
        self,
        embedding_model_name,
        chroma_db_path,
        chroma_db_collection_name,
        query_collection_name,
    ) -> np.ndarray:
        self.embedding_model_name = embedding_model_name
        self.chroma_db_path = chroma_db_path
        self.chroma_db_collection_name = chroma_db_collection_name
        self.query_collection_name = query_collection_name
        self.df = None
        self.risk_embeddings = None
        self.chroma_client = chromadb.PersistentClient(self.chroma_db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.chroma_db_collection_name
        )
        self.query_collection = self.chroma_client.get_or_create_collection(
            name=self.query_collection_name
        )

    def generate_embedding(self, query: str) -> np.ndarray:
        challenge = ollama.embed(model=self.embedding_model_name, input=query)
        query_embed = challenge["embeddings"][0]

        return np.array(query_embed)

    def data_load(self, file_path) -> pd.DataFrame:
        self.df = pd.read_csv(file_path)
        return self.df

    def create_search_text_column(self):

        if self.df is None:
            raise ValueError("Data is not loaded. Please load data")

        self.df["search_text"] = self.df.apply(self.create_row_text, axis=1)

        return self.df

    def create_row_text(self, row) -> str:
        intent = f"""
        This is a {row["risk_type"]} risk for the {row["project_name"]} construction project.
        The risk has a {row["severity_level"]} severity and is {row["likelihood"]} likely to occur.
        If the risk occurs, it may cause approximately {row["impact_days"]} days of impact
        and a financial impact of USD {row["financial_impact_usd"]}.
        The current status is {row["status"]}.
        The current mitigation plan is {row["mitigation_plan"]}.
        The risk was reported in the {row["region"]} region.
        The associated tags are {row["tags"]}.
        """.strip()

        raw_data = f"""
        Risk ID: {row["risk_id"]}
        Project ID: {row["project_id"]}
        Project Name: {row["project_name"]}
        Date Reported: {row["date_reported"]}
        Risk Type: {row["risk_type"]}
        Description: {row["description"]}
        Severity Level: {row["severity_level"]}
        Likelihood: {row["likelihood"]}
        Impact Days: {row["impact_days"]}
        Financial Impact USD: {row["financial_impact_usd"]}
        Status: {row["status"]}
        Mitigation Plan: {row["mitigation_plan"]}
        Reported By: {row["reported_by"]}
        Region: {row["region"]}
        Tags: {row["tags"]}
        """.strip()

        return intent + "\n\n" + raw_data

    def index_risks_in_chroma_db(self):
        if self.df is None:
            raise ValueError("Data is not loaded. Please load data")

        if "search_text" not in self.df.columns:
            raise ValueError(
                "Search text column is not created. Please create search text column"
            )

        for index, row in self.df.iterrows():
            risk_id = row["risk_id"]
            search_text = row["search_text"]
            embedding = self.generate_embedding(search_text)
            self.collection.upsert(
                ids=[str(risk_id)],
                embeddings=[embedding],
                documents=[search_text],
                metadatas=[{"risk_id": risk_id}],
            )

    def query_chroma_db(self, query: str, top_k: int = 1000):
        query_embedding = self.generate_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        return results

    def summarize_risks_1(self, context, user_query):

        prompt = f"""
        You are a Construction Project Risk Analysis Assistant.

        The user asked:

        {user_query}

        The following data was retrieved from the construction
        risk database using semantic similarity search.

        --------------------
        RETRIEVED DATA
        --------------------

        {context}

        --------------------
        YOUR TASK
        --------------------

        Answer the user's question directly using the retrieved data.

        IMPORTANT:

        1. First understand exactly what the user is asking for.
        2. Answer ONLY what the user requested.
        3. Do not automatically summarize all risks.
        4. Do not provide severity, likelihood, impact, financial impact,
        mitigation plans, or other information unless relevant to the
        user's question.
        5. If the user asks for a list, return a list.
        6. If the user asks for project names, return project names.
        7. If multiple retrieved records belong to the same project,
        list that project only once.
        8. If the user asks for a count, provide the count.
        9. If the user asks for a summary, provide a summary.
        10. Use ONLY the retrieved data.
        11. Do not invent information.
        12. Do not create numerical values that are not provided.
        13. If the requested information is not available in the retrieved
        data, say "No matching information was found."

        --------------------
        RESPONSE FORMAT
        --------------------

        Choose the response format based on the user's question.

        For a list request:
        - Return a clean numbered or bulleted list.

        For a count request:
        - Return the count and briefly explain what was counted.

        For a detail request:
        - Return the requested details.

        For a summary request:
        - Provide a concise summary.

        For a comparison request:
        - Present the relevant comparison.

        Do not add unrelated information.
        """

        response = ollama.chat(
            model="llama3.2", messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]

    def summarize_risks(self, user_query, context):

        prompt = f"""
    You are a Construction Project Risk Analysis Assistant.

    The user asked:

    {user_query}

    The following risks were retrieved from the construction
    risk database using semantic similarity search.

    --------------------
    RETRIEVED RISKS
    --------------------

    {context}

    --------------------
    INSTRUCTIONS
    --------------------

    Summarize the retrieved risks for the user.

    Include:

    1. Overall summary
    2. Most important risks
    3. Affected projects
    4. Severity and likelihood
    5. Potential impact
    6. Financial impact when available
    7. Current mitigation plans
    8. Common patterns or themes

    Important rules:

    - Use ONLY the information provided in the retrieved risks.
    - Do not invent facts.
    - Do not create numerical values that are not provided.
    - If information is missing, say "Not available".
    - Clearly distinguish individual risks from overall observations.
    - Keep the response concise and easy to understand.
    """

        response = ollama.chat(
            model="llama3.2", messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]

    def prepare_context(self, results):

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        context_parts = []

        for index, (document, metadata) in enumerate(
            zip(documents, metadatas), start=1
        ):
            risk_context = f"""
            Risk {index}
            --------------------------------
            Risk ID: {metadata.get("risk_id")}
            Project ID: {metadata.get("project_id")}
            Project Name: {metadata.get("project_name")}
            Date Reported: {metadata.get("date_reported")}

            Risk Type: {metadata.get("risk_type")}
            Description: {document}

            Severity Level: {metadata.get("severity_level")}
            Likelihood: {metadata.get("likelihood")}
            Impact Days: {metadata.get("impact_days")}
            Financial Impact USD: {metadata.get("financial_impact_usd")}

            Status: {metadata.get("status")}
            Mitigation Plan: {metadata.get("mitigation_plan")}

            Reported By: {metadata.get("reported_by")}
            Region: {metadata.get("region")}
            Tags: {metadata.get("tags")}
            """

            context_parts.append(risk_context)

        return "\n".join(context_parts)

    def save_text_to_chroma_db(self):

        query_knowledge = [
            """
            Column: project_name

            Meaning:
            Name of the construction project.

            Data type:
            string

            Example:
            Skyline Commercial Tower

            Common operations:
            SELECT, FILTER, DISTINCT, GROUP_BY
            """,
            """
            Column: region

            Meaning:
            Geographical region of the construction project.

            Data type:
            string

            Example values:
            North, South, East, West

            Common operations:
            FILTER, GROUP_BY, COUNT
            """,
            """
            Column: risk_type

            Meaning:
            Category of the construction risk.

            Example values:
            Schedule, Cost, Safety, Resource,
            Environmental, Compliance

            Common operations:
            FILTER, GROUP_BY, COUNT
            """,
            """
            Column: severity_level

            Meaning:
            Severity of the identified risk.

            Example values:
            Low, Medium, High, Critical

            Common operations:
            FILTER, GROUP_BY, COUNT
            """,
            """
            Column: likelihood

            Meaning:
            Probability of the risk occurring.

            Example values:
            Low, Medium, High

            Common operations:
            FILTER, GROUP_BY, COUNT
            """,
            """
            Column: impact_days

            Meaning:
            Number of project days potentially affected by the risk.

            Data type:
            numeric

            Common operations:
            SUM, AVERAGE, MIN, MAX, SORT, TOP_N
            """,
            """
            Column: financial_impact_usd

            Meaning:
            Estimated financial impact of the risk in USD.

            Data type:
            numeric

            Common operations:
            SUM, AVERAGE, MIN, MAX, SORT, TOP_N
            """,
            """
            Column: status

            Meaning:
            Current status of the risk.

            Common operations:
            FILTER, GROUP_BY, COUNT
            """,
            """
            Operation: FILTER

            Meaning:
            Select records that match one or more conditions.

            Example:
            Find risks in the North region.

            Typical columns:
            region, risk_type, severity_level, likelihood, status
            """,
            """
            Operation: SELECT

            Meaning:
            Return specific columns from the DataFrame.

            Example:
            Show project names.

            Typical columns:
            project_name, project_id, risk_type
            """,
            """
            Operation: DISTINCT

            Meaning:
            Return unique values without duplicates.

            Example:
            List unique project names.

            Typical columns:
            project_name, project_id, region, risk_type
            """,
            """
            Operation: GROUP_BY

            Meaning:
            Group records by one or more columns.

            Example:
            Show the number of risks by region.

            Typical columns:
            region, risk_type, severity_level, status
            """,
            """
            Operation: TOP_N

            Meaning:
            Return the highest or lowest N records according
            to a numeric or sortable column.

            Example:
            Show the top 3 risks by financial impact.

            Typical columns:
            financial_impact_usd, impact_days
            """,
            """
            Operation: SORT

            Meaning:
            Order records according to a column.

            Example:
            Sort risks by financial impact.

            Typical columns:
            financial_impact_usd, impact_days,
            date_reported, severity_level
            """,
        ]

        for index, text in enumerate(query_knowledge):
            embedding = self.generate_embedding(text)

            self.query_collection.upsert(
                ids=[f"query_knowledge_{index}"],
                embeddings=[embedding],
                documents=[text],
                metadatas=[
                    {
                        "knowledge_type": "query_knowledge",
                        "knowledge_id": f"query_knowledge_{index}",
                    }
                ],
            )

        print(f"Indexed {len(query_knowledge)} query knowledge documents.")

    def query_query_knowledge(self, query: str, top_k: int = 5):

        query_embedding = self.generate_embedding(query)

        results = self.query_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"knowledge_type": "query_knowledge"},
        )

        return results

    def prepare_query_knowledge_context(self, results):

        documents = results["documents"][0]

        context = ""

        for index, document in enumerate(documents, start=1):
            context += f"""
            Knowledge {index}
            -------------------------
            {document}

        """

        return context

        prompt = f"""
        You are a Construction Risk Data Query Planner.

        Your ONLY job is to convert the user's natural-language
        question into a structured JSON query plan for a Pandas DataFrame.

        You MUST NOT answer the user's question.

        You MUST NOT generate Python code.

        You MUST NOT generate SQL.

        You MUST return ONLY valid JSON.

        ==================================================
        USER QUESTION
        ==================================================

        {user_query}

        ==================================================
        QUERY KNOWLEDGE FROM RAG
        ==================================================

        {rag_context}

        ==================================================
        AVAILABLE DATAFRAME COLUMNS
        ==================================================

        risk_id
        project_id
        project_name
        date_reported
        risk_type
        description
        severity_level
        likelihood
        impact_days
        financial_impact_usd
        status
        mitigation_plan
        reported_by
        region
        tags

        ==================================================
        SUPPORTED OPERATIONS
        ==================================================

        FILTER
        SELECT
        DISTINCT
        COUNT
        SUM
        AVERAGE
        MIN
        MAX
        SORT
        TOP_N
        GROUP_BY
        AGGREGATE
        FILTER_SELECT
        FILTER_GROUP

        ==================================================
        SUPPORTED FILTER OPERATORS
        ==================================================

        equals
        not_equals
        in
        not_in
        contains
        greater_than
        less_than
        greater_than_or_equal
        less_than_or_equal

        ==================================================
        MANDATORY FILTER STRUCTURE
        ==================================================

        EVERY filter MUST contain EXACTLY these three properties:

        1. column
        2. operator
        3. value

        For example:

        {{
            "column": "region",
            "operator": "equals",
            "value": "North"
        }}

        NEVER generate this:

        {{
            "column": "region",
            "value": "North"
        }}

        The "operator" property is mandatory.

        ==================================================
        IMPORTANT MAPPING RULES
        ==================================================

        If the user says:

        "North region"

        generate:

        {{
            "column": "region",
            "operator": "equals",
            "value": "North"
        }}

        If the user says:

        "Critical risks"

        generate:

        {{
            "column": "severity_level",
            "operator": "equals",
            "value": "Critical"
        }}

        If the user says:

        "Safety risks"

        generate:

        {{
            "column": "risk_type",
            "operator": "equals",
            "value": "Safety"
        }}

        If the user says:

        "financial impact greater than 100000"

        generate:

        {{
            "column": "financial_impact_usd",
            "operator": "greater_than",
            "value": 100000
        }}

        If the user says:

        "impact days greater than 30"

        generate:

        {{
            "column": "impact_days",
            "operator": "greater_than",
            "value": 30
        }}

        ==================================================
        QUERY RULES
        ==================================================

        1. Use ONLY available DataFrame columns.

        2. Use ONLY supported operations.

        3. Use ONLY supported filter operators.

        4. Every filter MUST contain:
        - column
        - operator
        - value

        5. Never omit the operator.

        6. If the user asks for project names,
        use "project_name".

        7. If the user asks for unique project names,
        set "distinct": true.

        8. If the user asks for a specific region,
        filter using "region" and operator "equals".

        9. If the user asks "how many",
        use COUNT.

        10. If the user asks "total",
        use SUM where appropriate.

        11. If the user asks "average",
        use AVERAGE.

        12. If the user asks for top N,
        use TOP_N.

        13. If the user asks for results by a category,
        use GROUP_BY or AGGREGATE.

        14. Do not add filters that the user did not request.

        15. Do not invent values.

        16. Do not invent column names.

        17. Do not explain the query.

        18. Return ONLY JSON.

        ==================================================
        COUNT RULES
        ==================================================

        If the user asks:

        "count the number of projects"

        you MUST use:

        "operation": "COUNT"

        and:

        "column": "project_id"

        and:

        "distinct": true

        Do NOT put "count(*)" inside the "select" array.

        "count(*)" is NOT a DataFrame column.

        For example:

        User:
        count the number of projects in west region

        Correct:

        {
            "query_type": "STRUCTURED",
            "operation": "COUNT",
            "filters": [
                {
                "column": "region",
                    "operator": "equals",
                    "value": "West"
                }
            ],
            "column": "project_id",
            "distinct": true
        }

        Incorrect:

        {
            "operation": "GROUP_BY",
            "select": ["count(*)"]
        }

        ==================================================
        COUNT EXAMPLES
        ==================================================

        Question:
        How many projects are in North region?

        Return:

        {
            "query_type": "STRUCTURED",
            "operation": "COUNT",
            "filters": [
                {
                "column": "region",
                    "operator": "equals",
                    "value": "North"
                }
            ],
            "column": "project_id",
            "distinct": true
        }

        Question:
        How many risks are in North region?

        Return:

        {
            "query_type": "STRUCTURED",
            "operation": "COUNT",
            "filters": [
                {
                "column": "region",
                    "operator": "equals",
                    "value": "North"
                }
            ],
            "column": "risk_id",
            "distinct": false
        }

        ==================================================
        EXACT OUTPUT EXAMPLE
        ==================================================

        For this user question:

        "Give me the list of project names in North region"

        you MUST return:

        {{
            "query_type": "STRUCTURED",
            "operation": "FILTER_SELECT",
            "filters": [
                {{
                    "column": "region",
                    "operator": "equals",
                    "value": "North"
                }}
            ],
            "select": [
                "project_name"
            ],
            "distinct": true
        }}

        ==================================================
        ANOTHER EXAMPLE
        ==================================================

        User question:

        "Show Critical safety risks in South region"

        Return:

        {{
            "query_type": "STRUCTURED",
            "operation": "FILTER_SELECT",
            "filters": [
                {{
                    "column": "severity_level",
                    "operator": "equals",
                    "value": "Critical"
                }},
                {{
                    "column": "risk_type",
                    "operator": "equals",
                    "value": "Safety"
                }},
                {{
                    "column": "region",
                    "operator": "equals",
                    "value": "South"
                }}
            ],
            "select": [
                "risk_id",
                "project_name",
                "description"
            ],
            "distinct": false
        }}

        ==================================================
        FINAL VALIDATION BEFORE RESPONDING
        ==================================================

        Before returning the JSON, verify:

        - Is operation present?
        - Is filters a list?
        - Does EVERY filter contain "column"?
        - Does EVERY filter contain "operator"?
        - Does EVERY filter contain "value"?
        - Are all columns valid?
        - Are all operators valid?
        - Is the JSON valid?

        If any filter does not contain an operator,
        ADD:

        "operator": "equals"

        when the user is requesting an exact value.

        Return ONLY valid JSON.
        """

        response = ollama.chat(
            model="llama3.2", messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]

    def execute_query_plan(self, query_plan):
        """
        Execute an LLM-generated structured query plan
        against the Pandas DataFrame.

        query_plan can be either:
        1. A Python dictionary
        2. A JSON string returned by the LLM
        """

        if self.df is None:
            raise ValueError("Data is not loaded.")

        # =====================================================
        # STEP 1: Convert JSON string to dictionary
        # =====================================================

        if isinstance(query_plan, str):
            # Remove markdown code fences if LLM returned them
            query_plan = query_plan.strip()

            if query_plan.startswith("```json"):
                query_plan = query_plan[7:]

            elif query_plan.startswith("```"):
                query_plan = query_plan[3:]

            if query_plan.endswith("```"):
                query_plan = query_plan[:-3]

            query_plan = query_plan.strip()

            try:
                query_plan = json.loads(query_plan)

            except json.JSONDecodeError as ex:
                raise ValueError(
                    f"Invalid query plan JSON: {ex}\n\nQuery Plan:\n{query_plan}"
                )

        # =====================================================
        # STEP 2: Validate query plan
        # =====================================================

        if not isinstance(query_plan, dict):
            raise ValueError("Query plan must be a dictionary or valid JSON object.")

        operation = query_plan.get("operation")

        if not operation:
            raise ValueError("Query plan is missing 'operation'.")

        # =====================================================
        # STEP 3: Copy DataFrame
        # =====================================================

        df = self.df.copy()

        available_columns = set(df.columns)

        # =====================================================
        # STEP 4: Validate filters
        # =====================================================

        filters = query_plan.get("filters", [])

        if not isinstance(filters, list):
            raise ValueError("'filters' must be a list.")

        for condition in filters:
            if not isinstance(condition, dict):
                raise ValueError(f"Invalid filter condition: {condition}")

            column = condition.get("column")
            operator = condition.get("operator")
            value = condition.get("value")

            if not column:
                raise ValueError(f"Filter is missing 'column': {condition}")

            if column not in available_columns:
                raise ValueError(f"Invalid filter column: {column}")

            if not operator:
                # If operator is missing, default to equals.
                #
                # This is safe for simple exact-value filters such as:
                # region = North
                # severity_level = Critical
                # risk_type = Safety

                operator = "equals"
                condition["operator"] = operator

            if "value" not in condition:
                raise ValueError(f"Filter is missing 'value': {condition}")

        # =====================================================
        # STEP 5: Validate SELECT columns
        # =====================================================

        select_columns = query_plan.get("select", [])

        if not isinstance(select_columns, list):
            raise ValueError("'select' must be a list.")

        for column in select_columns:
            if column not in available_columns:
                raise ValueError(f"Invalid select column: {column}")

        # =====================================================
        # STEP 6: Apply FILTER
        # =====================================================

        for condition in filters:
            column = condition["column"]
            operator = condition["operator"]
            value = condition["value"]

            if operator == "equals":
                df = df[df[column] == value]

            elif operator == "not_equals":
                df = df[df[column] != value]

            elif operator == "in":
                if not isinstance(value, list):
                    raise ValueError(f"'in' operator requires a list: {value}")

                df = df[df[column].isin(value)]

            elif operator == "not_in":
                if not isinstance(value, list):
                    raise ValueError(f"'not_in' operator requires a list: {value}")

                df = df[~df[column].isin(value)]

            elif operator == "contains":
                df = df[
                    df[column]
                    .astype(str)
                    .str.contains(str(value), case=False, na=False)
                ]

            elif operator == "greater_than":
                df = df[df[column] > value]

            elif operator == "less_than":
                df = df[df[column] < value]

            elif operator == "greater_than_or_equal":
                df = df[df[column] >= value]

            elif operator == "less_than_or_equal":
                df = df[df[column] <= value]

            else:
                raise ValueError(f"Unsupported operator: {operator}")

        # =====================================================
        # STEP 7: Execute operation
        # =====================================================

        # -----------------------------------------------------
        # FILTER / SELECT / DISTINCT
        # -----------------------------------------------------

        if operation in ["FILTER", "SELECT", "DISTINCT", "FILTER_SELECT"]:
            if select_columns:
                df = df[select_columns]

            if query_plan.get("distinct", False):
                df = df.drop_duplicates()

        # -----------------------------------------------------
        # COUNT
        # -----------------------------------------------------

        # -----------------------------------------------------
        # COUNT
        # -----------------------------------------------------

        elif operation == "COUNT":
            count_column = query_plan.get("column")

            if not count_column:
                raise ValueError("COUNT operation requires 'column'.")

            if count_column not in available_columns:
                raise ValueError(f"Invalid COUNT column: {count_column}")

            distinct = query_plan.get("distinct", False)

            if distinct:
                return df[count_column].dropna().nunique()

            return df[count_column].dropna().count()

        # -----------------------------------------------------
        # SUM
        # -----------------------------------------------------

        elif operation == "SUM":
            column = query_plan.get("column")

            if not column:
                raise ValueError("SUM operation requires 'column'.")

            if column not in available_columns:
                raise ValueError(f"Invalid SUM column: {column}")

            return df[column].sum()

        # -----------------------------------------------------
        # AVERAGE
        # -----------------------------------------------------

        elif operation == "AVERAGE":
            column = query_plan.get("column")

            if not column:
                raise ValueError("AVERAGE operation requires 'column'.")

            if column not in available_columns:
                raise ValueError(f"Invalid AVERAGE column: {column}")

            return df[column].mean()

        # -----------------------------------------------------
        # MIN
        # -----------------------------------------------------

        elif operation == "MIN":
            column = query_plan.get("column")

            if not column:
                raise ValueError("MIN operation requires 'column'.")

            if column not in available_columns:
                raise ValueError(f"Invalid MIN column: {column}")

            return df[column].min()

        # -----------------------------------------------------
        # MAX
        # -----------------------------------------------------

        elif operation == "MAX":
            column = query_plan.get("column")

            if not column:
                raise ValueError("MAX operation requires 'column'.")

            if column not in available_columns:
                raise ValueError(f"Invalid MAX column: {column}")

            return df[column].max()

        # -----------------------------------------------------
        # SORT
        # -----------------------------------------------------

        elif operation == "SORT":
            column = query_plan.get("column")

            if not column:
                raise ValueError("SORT operation requires 'column'.")

            if column not in available_columns:
                raise ValueError(f"Invalid SORT column: {column}")

            order = query_plan.get("order", "ascending")

            ascending = order.lower() != "descending"

            df = df.sort_values(by=column, ascending=ascending)

        # -----------------------------------------------------
        # TOP_N
        # -----------------------------------------------------

        elif operation == "TOP_N":
            column = query_plan.get("column")

            if not column:
                raise ValueError("TOP_N operation requires 'column'.")

            if column not in available_columns:
                raise ValueError(f"Invalid TOP_N column: {column}")

            limit = query_plan.get("limit", 5)

            order = query_plan.get("order", "descending")

            ascending = order.lower() == "ascending"

            df = df.sort_values(by=column, ascending=ascending).head(limit)

            # Apply SELECT after TOP_N if requested

            if select_columns:
                df = df[select_columns]

        # -----------------------------------------------------
        # GROUP BY / AGGREGATE
        # -----------------------------------------------------

        elif operation in ["GROUP_BY", "AGGREGATE", "FILTER_GROUP"]:
            group_by = query_plan.get("group_by", [])

            if not group_by:
                raise ValueError("AGGREGATE requires 'group_by'.")

            if isinstance(group_by, str):
                group_by = [group_by]

            # -----------------------------------------
            # Validate GROUP BY columns
            # -----------------------------------------

            for column in group_by:
                if column not in available_columns:
                    raise ValueError(f"Invalid GROUP BY column: {column}")

            # -----------------------------------------
            # Get aggregations
            # -----------------------------------------

            aggregations = query_plan.get("aggregations", [])

            if not aggregations:
                raise ValueError("AGGREGATE requires 'aggregations'.")

            # -----------------------------------------
            # Build aggregation expressions
            # -----------------------------------------

            named_aggregations = {}

            for aggregation in aggregations:
                column = aggregation.get("column")

                function = aggregation.get("function")

                alias = aggregation.get("alias")

                if not column:
                    raise ValueError(f"Aggregation is missing column: {aggregation}")

                if not function:
                    raise ValueError(f"Aggregation is missing function: {aggregation}")

                if column not in available_columns:
                    raise ValueError(f"Invalid aggregation column: {column}")

                # Normalize function name

                function = function.lower()

                if function == "average":
                    function = "mean"

                elif function == "avg":
                    function = "mean"

                elif function == "count":
                    function = "count"

                elif function == "sum":
                    function = "sum"

                elif function == "min":
                    function = "min"

                elif function == "max":
                    function = "max"

                elif function == "unique_count":
                    function = "nunique"

                else:
                    raise ValueError(f"Unsupported aggregation function: {function}")

                if not alias:
                    alias = f"{function}_{column}"

                named_aggregations[alias] = (column, function)

            # -----------------------------------------
            # Execute GROUP BY
            # -----------------------------------------

            result = df.groupby(group_by).agg(**named_aggregations).reset_index()

            return result

        return df

    def generate_pandas_query(self, user_query, rag_context):
        """
        Convert a natural-language construction-risk question into a validated
        structured JSON query plan for the Pandas executor.

        Important design:
        - JSON examples are created with Python dictionaries + json.dumps().
        This prevents f-string brace errors.
        - The LLM is instructed to return JSON only.
        - The returned JSON is parsed and normalized before it is returned.
        - Grouping phrases such as "each region" can never become filter values.
        - GROUP BY + aggregation always uses the canonical AGGREGATE operation.
        """

        # =========================================================
        # DATAFRAME SCHEMA
        # =========================================================

        valid_columns = {
            "risk_id",
            "project_id",
            "project_name",
            "date_reported",
            "risk_type",
            "description",
            "severity_level",
            "likelihood",
            "impact_days",
            "financial_impact_usd",
            "status",
            "mitigation_plan",
            "reported_by",
            "region",
            "tags",
        }

        valid_operations = {
            "FILTER",
            "SELECT",
            "DISTINCT",
            "COUNT",
            "SUM",
            "AVERAGE",
            "MIN",
            "MAX",
            "SORT",
            "TOP_N",
            "AGGREGATE",
            "FILTER_SELECT",
            "FILTER_GROUP",
        }

        valid_filter_operators = {
            "equals",
            "not_equals",
            "in",
            "not_in",
            "contains",
            "greater_than",
            "less_than",
            "greater_than_or_equal",
            "less_than_or_equal",
        }

        valid_aggregation_functions = {
            "count",
            "sum",
            "average",
            "min",
            "max",
        }

        # =========================================================
        # JSON EXAMPLES
        # =========================================================

        filter_example = {
            "column": "region",
            "operator": "equals",
            "value": "North",
        }

        count_risks_example = {
            "query_type": "STRUCTURED",
            "operation": "COUNT",
            "filters": [
                {
                    "column": "region",
                    "operator": "equals",
                    "value": "West",
                }
            ],
            "column": "risk_id",
            "distinct": False,
        }

        count_projects_example = {
            "query_type": "STRUCTURED",
            "operation": "COUNT",
            "filters": [
                {
                    "column": "region",
                    "operator": "equals",
                    "value": "West",
                }
            ],
            "column": "project_id",
            "distinct": True,
        }

        group_by_region_example = {
            "query_type": "STRUCTURED",
            "operation": "AGGREGATE",
            "filters": [],
            "group_by": ["region"],
            "aggregations": [
                {
                    "column": "risk_id",
                    "function": "count",
                    "alias": "risk_count",
                }
            ],
        }

        group_by_financial_example = {
            "query_type": "STRUCTURED",
            "operation": "AGGREGATE",
            "filters": [],
            "group_by": ["region"],
            "aggregations": [
                {
                    "column": "financial_impact_usd",
                    "function": "sum",
                    "alias": "total_financial_impact",
                }
            ],
        }

        multiple_aggregation_example = {
            "query_type": "STRUCTURED",
            "operation": "AGGREGATE",
            "filters": [],
            "group_by": ["region"],
            "aggregations": [
                {
                    "column": "risk_id",
                    "function": "count",
                    "alias": "risk_count",
                },
                {
                    "column": "financial_impact_usd",
                    "function": "sum",
                    "alias": "total_financial_impact",
                },
            ],
        }

        filtered_group_example = {
            "query_type": "STRUCTURED",
            "operation": "AGGREGATE",
            "filters": [
                {
                    "column": "severity_level",
                    "operator": "equals",
                    "value": "Critical",
                }
            ],
            "group_by": ["region"],
            "aggregations": [
                {
                    "column": "risk_id",
                    "function": "count",
                    "alias": "critical_risk_count",
                }
            ],
        }

        # Negative/positive grouping examples are generated safely.
        wrong_group_filter_example = {
            "filters": [
                {
                    "column": "region",
                    "operator": "equals",
                    "value": "each region",
                }
            ],
            "group_by": ["region"],
        }

        correct_group_filter_example = {
            "filters": [],
            "group_by": ["region"],
        }

        # =========================================================
        # SERIALIZE EXAMPLES
        # =========================================================

        def as_json(value):
            return json.dumps(value, indent=4)

        filter_example_json = as_json(filter_example)
        count_risks_example_json = as_json(count_risks_example)
        count_projects_example_json = as_json(count_projects_example)
        group_by_region_example_json = as_json(group_by_region_example)
        group_by_financial_example_json = as_json(group_by_financial_example)
        multiple_aggregation_example_json = as_json(multiple_aggregation_example)
        filtered_group_example_json = as_json(filtered_group_example)
        wrong_group_filter_example_json = as_json(wrong_group_filter_example)
        correct_group_filter_example_json = as_json(correct_group_filter_example)

        # =========================================================
        # PROMPT
        # =========================================================

        prompt = f"""
    You are a Construction Risk Data Query Planner.

    Your ONLY job is to convert the user's natural-language
    question into ONE structured JSON query plan for a Pandas
    DataFrame.

    You MUST NOT answer the question.
    You MUST NOT generate Python.
    You MUST NOT generate SQL.
    You MUST NOT calculate the result.
    You MUST return ONLY valid JSON.

    =========================================================
    USER QUESTION
    =========================================================

    {user_query}

    =========================================================
    QUERY KNOWLEDGE FROM RAG
    =========================================================

    {rag_context}

    =========================================================
    AVAILABLE DATAFRAME COLUMNS
    =========================================================

    risk_id
    project_id
    project_name
    date_reported
    risk_type
    description
    severity_level
    likelihood
    impact_days
    financial_impact_usd
    status
    mitigation_plan
    reported_by
    region
    tags

    =========================================================
    SUPPORTED OPERATIONS
    =========================================================

    FILTER
    SELECT
    DISTINCT
    COUNT
    SUM
    AVERAGE
    MIN
    MAX
    SORT
    TOP_N
    AGGREGATE
    FILTER_SELECT
    FILTER_GROUP

    IMPORTANT:
    GROUP_BY is NOT a valid final operation.
    For GROUP BY + aggregation, ALWAYS use AGGREGATE.

    =========================================================
    FILTER STRUCTURE
    =========================================================

    Every filter MUST contain:

    column
    operator
    value

    Example:

    {filter_example_json}

    If the user gives an exact value, use:
    operator = "equals".

    Never omit the operator.

    =========================================================
    FILTER OPERATORS
    =========================================================

    equals
    not_equals
    in
    not_in
    contains
    greater_than
    less_than
    greater_than_or_equal
    less_than_or_equal

    =========================================================
    FILTER RULE
    =========================================================

    Create a filter ONLY when the user explicitly specifies
    a condition.

    Examples:

    "North region"
        -> region = North

    "West region"
        -> region = West

    "Critical risks"
        -> severity_level = Critical

    "Safety risks"
        -> risk_type = Safety

    DO NOT invent filters.

    =========================================================
    CRITICAL GROUPING RULE
    =========================================================

    The following phrases indicate GROUPING:

    "each"
    "every"
    "by"
    "per"
    "for each"

    When these phrases introduce a category, they are NOT
    filter values.

    Examples:

    "each region"
    "every region"
    "by region"
    "per region"
    "for each region"

    These mean:

    group_by = ["region"]

    They NEVER mean:

    region = "each region"
    region = "every region"
    region = "by region"
    region = "per region"
    region = "for each region"

    =========================================================
    VERY IMPORTANT NEGATIVE EXAMPLE
    =========================================================

    For:

    "How many risks are there in each region?"

    NEVER generate this:

    {wrong_group_filter_example_json}

    This is WRONG because "each region" is a grouping
    instruction, not a region value.

    The correct interpretation is:

    {correct_group_filter_example_json}

    =========================================================
    GROUPING HAS PRIORITY
    =========================================================

    If the user asks:

    "How many risks are there in each region?"

    interpret it as:

    1. No region filter.
    2. Group by region.
    3. Count risk_id.

    Therefore:

    filters = []

    group_by = ["region"]

    operation = "AGGREGATE"

    =========================================================
    FILTER VS GROUPING
    =========================================================

    Question:

    "How many risks are there in West region?"

    This is a FILTER.

    Use:

    {count_risks_example_json}

    Question:

    "How many risks are there in each region?"

    This is GROUPING.

    Use:

    {group_by_region_example_json}

    Do NOT convert "each region" into a filter.

    =========================================================
    COUNT RULES
    =========================================================

    ONE overall risk count:

    operation = COUNT
    column = risk_id
    distinct = false

    ONE overall project count:

    operation = COUNT
    column = project_id
    distinct = true

    Why?
    One project can have multiple risk records.

    For grouped counts, use AGGREGATE.

    Example:

    "How many risks are there in each region?"

    Use:

    {group_by_region_example_json}

    Example:

    "How many projects are there in each region?"

    Use:

    operation = AGGREGATE
    group_by = ["region"]

    aggregation:

    column = project_id
    function = count
    alias = project_count

    For project counts, use distinct project IDs if the
    executor supports distinct aggregation. If the executor
    does not support distinct aggregation, do not invent a
    new aggregation function; use the supported schema.

    =========================================================
    NEVER USE count(*)
    =========================================================

    Never generate:

    count(*)

    Never use count(*) as a DataFrame column.

    For risk counts use:

    column = risk_id
    function = count

    =========================================================
    GROUP BY + COUNT
    =========================================================

    "How many risks are there in each region?"

    Use:

    {group_by_region_example_json}

    "How many risks are there for each risk type?"

    Use:

    group_by = ["risk_type"]

    aggregation:

    column = risk_id
    function = count
    alias = risk_count

    "How many risks are there per severity level?"

    Use:

    group_by = ["severity_level"]

    aggregation:

    column = risk_id
    function = count
    alias = risk_count

    =========================================================
    GROUP BY + SUM
    =========================================================

    "Show total financial impact by region."

    Use:

    {group_by_financial_example_json}

    =========================================================
    GROUP BY + MULTIPLE AGGREGATIONS
    =========================================================

    "For each region, show the number of risks and total
    financial impact."

    Use:

    {multiple_aggregation_example_json}

    =========================================================
    FILTER + GROUP BY
    =========================================================

    Filters and grouping may coexist.

    Example:

    "Show the number of Critical risks by region."

    Use:

    {filtered_group_example_json}

    Meaning:

    1. Filter severity_level = Critical.
    2. Group by region.
    3. Count risk_id.

    IMPORTANT:
    Do NOT add region = "by region" or
    region = "each region".

    =========================================================
    AGGREGATION FUNCTIONS
    =========================================================

    Supported functions:

    count
    sum
    average
    min
    max

    Always use lowercase.

    =========================================================
    SUM
    =========================================================

    "What is the total financial impact?"

    Use:

    operation = SUM
    column = financial_impact_usd

    =========================================================
    AVERAGE
    =========================================================

    "What is the average impact days?"

    Use:

    operation = AVERAGE
    column = impact_days

    =========================================================
    MIN / MAX
    =========================================================

    "What is the minimum impact days?"

    Use:

    operation = MIN
    column = impact_days

    "What is the maximum impact days?"

    Use:

    operation = MAX
    column = impact_days

    =========================================================
    TOP N
    =========================================================

    "Show the top 3 risks by financial impact."

    Use:

    operation = TOP_N
    column = financial_impact_usd
    order = descending
    limit = 3

    =========================================================
    SORT
    =========================================================

    "Sort risks by financial impact."

    Use:

    operation = SORT
    column = financial_impact_usd

    Do not invent filters.

    =========================================================
    SELECT
    =========================================================

    "Show project names."

    Use:

    operation = SELECT
    select = ["project_name"]

    "Show project names in North region."

    Use:

    operation = FILTER_SELECT

    filters:

    region = North

    select:

    ["project_name"]

    =========================================================
    DISTINCT
    =========================================================

    "List unique project names."

    Use:

    operation = DISTINCT
    select = ["project_name"]

    or the equivalent SELECT + distinct structure supported
    by the executor.

    For:

    "How many projects?"

    Use:

    operation = COUNT
    column = project_id
    distinct = true

    =========================================================
    VALID QUERY PLAN STRUCTURES
    =========================================================

    FILTER_SELECT:

    {{
        "query_type": "STRUCTURED",
        "operation": "FILTER_SELECT",
        "filters": [],
        "select": [],
        "distinct": false
    }}

    COUNT:

    {{
        "query_type": "STRUCTURED",
        "operation": "COUNT",
        "filters": [],
        "column": "risk_id",
        "distinct": false
    }}

    AGGREGATE:

    {{
        "query_type": "STRUCTURED",
        "operation": "AGGREGATE",
        "filters": [],
        "group_by": [],
        "aggregations": []
    }}

    TOP_N:

    {{
        "query_type": "STRUCTURED",
        "operation": "TOP_N",
        "filters": [],
        "column": "financial_impact_usd",
        "order": "descending",
        "limit": 3
    }}

    =========================================================
    STRICT RESTRICTIONS
    =========================================================

    1. Use ONLY listed DataFrame columns.
    2. Use ONLY listed operations.
    3. Use ONLY listed filter operators.
    4. Never invent a column.
    5. Never invent a filter value.
    6. Never create a filter from "each", "every", "by",
    "per", or "for each" when those words indicate grouping.
    7. Every filter MUST have column, operator, and value.
    8. Never use count(*).
    9. Never put expressions inside select.
    10. GROUP BY + aggregation MUST use AGGREGATE.
    11. Do not use GROUP_BY as the final operation.
    12. Do not generate Python.
    13. Do not generate SQL.
    14. Do not calculate the result.
    15. Do not return explanations.
    16. Return JSON only.
    17. If grouping is requested, group_by MUST contain the
        requested DataFrame column.
    18. If grouping is requested, filters MUST NOT contain
        the grouping phrase as a value.
    19. Do not invent North/South/East/West filters when the
        user says "each region".
    20. Preserve explicit filters while also grouping when
        the user specifies both.

    =========================================================
    FINAL VALIDATION BEFORE RESPONSE
    =========================================================

    Before returning JSON, verify:

    1. operation exists.
    2. operation is supported.
    3. filters is a list.
    4. Every filter has:
    column
    operator
    value
    5. Every filter column is valid.
    6. Every filter operator is valid.
    7. No filter value is a grouping phrase.
    8. If operation = AGGREGATE:
    group_by exists and is a list.
    9. If operation = AGGREGATE:
    aggregations exists and is a list.
    10. Every aggregation has:
        column
        function
        alias
    11. Every aggregation column is valid.
    12. Every aggregation function is valid.
    13. count(*) does not appear.
    14. GROUP_BY does not appear as operation.
    15. If question contains "each", "every", "by", "per",
        or "for each" and it means grouping:
        group_by is populated and no grouping phrase is used
        as a filter value.
    16. If the user explicitly specifies another condition,
        preserve that filter.
    17. Return valid JSON only.

    =========================================================
    FINAL RESPONSE
    =========================================================

    Return ONLY the JSON query plan.

    Do not use markdown code fences.
    Do not include explanations.
    Do not include commentary.
    """

        # =========================================================
        # CALL LLAMA
        # =========================================================

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        raw_content = response["message"]["content"].strip()

        # =========================================================
        # PARSE LLM JSON
        # =========================================================

        def extract_json_object(text):
            """
            Extract the first JSON object from the model response.
            Handles accidental markdown fences or surrounding text.
            """
            text = text.strip()

            # Remove common markdown fences.
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

            # First attempt: entire response is JSON.
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Second attempt: locate the outermost JSON object.
            start = text.find("{")
            if start == -1:
                raise ValueError("LLM did not return a JSON query plan.")

            depth = 0
            in_string = False
            escape = False

            for index in range(start, len(text)):
                char = text[index]

                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1

                    if depth == 0:
                        candidate = text[start : index + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"LLM returned invalid JSON: {exc}"
                            ) from exc

            raise ValueError(
                "Could not find a complete JSON object in the LLM response."
            )

        query_plan = extract_json_object(raw_content)

        if not isinstance(query_plan, dict):
            raise ValueError("Query plan must be a JSON object.")

        # =========================================================
        # NORMALIZATION
        # =========================================================

        operation = str(query_plan.get("operation", "")).strip().upper()

        # Backward compatibility:
        # If the LLM still returns GROUP_BY, convert it to AGGREGATE.
        if operation == "GROUP_BY":
            operation = "AGGREGATE"
            query_plan["operation"] = operation

        query_plan["query_type"] = "STRUCTURED"
        query_plan["operation"] = operation

        # Normalize missing filters.
        filters = query_plan.get("filters", [])

        if filters is None:
            filters = []

        if not isinstance(filters, list):
            raise ValueError("Query plan 'filters' must be a list.")

        # =========================================================
        # REMOVE INVALID GROUPING FILTERS
        # =========================================================

        grouping_patterns = {
            "each region",
            "every region",
            "by region",
            "per region",
            "for each region",
            "each risk type",
            "every risk type",
            "by risk type",
            "per risk type",
            "for each risk type",
            "each severity level",
            "every severity level",
            "by severity level",
            "per severity level",
            "for each severity level",
            "each status",
            "every status",
            "by status",
            "per status",
            "for each status",
            "each project",
            "every project",
            "by project",
            "per project",
            "for each project",
        }

        normalized_filters = []

        for condition in filters:
            if not isinstance(condition, dict):
                continue

            column = str(condition.get("column", "")).strip()

            operator = str(condition.get("operator", "")).strip().lower()

            value = condition.get("value")

            value_text = str(value).strip().lower() if value is not None else ""

            # Remove filters such as:
            # region = "each region"
            if value_text in grouping_patterns:
                continue

            # Missing operator: safe default for exact-value filters.
            if not operator:
                operator = "equals"

            condition["column"] = column
            condition["operator"] = operator

            normalized_filters.append(condition)

        query_plan["filters"] = normalized_filters

        # =========================================================
        # DETERMINE GROUPING FROM USER QUESTION
        # =========================================================

        question = str(user_query).strip().lower()

        grouping_phrases = [
            "each ",
            "every ",
            "by ",
            "per ",
            "for each ",
        ]

        has_grouping_language = any(phrase in question for phrase in grouping_phrases)

        # Map natural-language grouping phrases to known columns.
        grouping_column_patterns = [
            (r"\b(each|every|by|per)\s+region\b", "region"),
            (r"\bfor\s+each\s+region\b", "region"),
            (r"\b(each|every|by|per)\s+risk\s+type\b", "risk_type"),
            (r"\bfor\s+each\s+risk\s+type\b", "risk_type"),
            (
                r"\b(each|every|by|per)\s+severity\s+level\b",
                "severity_level",
            ),
            (
                r"\bfor\s+each\s+severity\s+level\b",
                "severity_level",
            ),
            (r"\b(each|every|by|per)\s+status\b", "status"),
            (r"\bfor\s+each\s+status\b", "status"),
            (r"\b(each|every|by|per)\s+project\b", "project_name"),
            (r"\bfor\s+each\s+project\b", "project_name"),
            (r"\b(each|every|by|per)\s+likelihood\b", "likelihood"),
            (r"\bfor\s+each\s+likelihood\b", "likelihood"),
        ]

        detected_group_columns = []

        for pattern, column in grouping_column_patterns:
            if re.search(pattern, question):
                if column not in detected_group_columns:
                    detected_group_columns.append(column)

        # If the question clearly indicates grouping and the LLM
        # detected it, normalize the final operation to AGGREGATE.
        if has_grouping_language and detected_group_columns:
            query_plan["operation"] = "AGGREGATE"

            existing_group_by = query_plan.get("group_by", [])

            if existing_group_by is None:
                existing_group_by = []

            if not isinstance(existing_group_by, list):
                existing_group_by = [existing_group_by]

            normalized_group_by = []

            for column in existing_group_by:
                column = str(column).strip()
                if column in valid_columns and column not in normalized_group_by:
                    normalized_group_by.append(column)

            # Prefer the grouping column explicitly detected from the
            # user's question.
            for column in detected_group_columns:
                if column not in normalized_group_by:
                    normalized_group_by.append(column)

            query_plan["group_by"] = normalized_group_by

            # Remove any filter whose value is a grouping phrase or
            # whose value is literally "each <group>".
            final_filters = []

            for condition in query_plan["filters"]:
                value = condition.get("value")
                value_text = str(value).strip().lower() if value is not None else ""

                is_grouping_value = any(
                    phrase in value_text
                    for phrase in (
                        "each ",
                        "every ",
                        "by ",
                        "per ",
                        "for each ",
                    )
                )

                if is_grouping_value:
                    continue

                final_filters.append(condition)

            query_plan["filters"] = final_filters

        # =========================================================
        # VALIDATE OPERATION
        # =========================================================

        if operation not in valid_operations:
            raise ValueError(
                f"Unsupported query operation returned by LLM: {operation}"
            )

        # =========================================================
        # VALIDATE FILTERS
        # =========================================================

        for condition in query_plan["filters"]:
            column = condition.get("column")
            operator = condition.get("operator")
            value = condition.get("value")

            if column not in valid_columns:
                raise ValueError(f"Invalid filter column: {column}")

            if operator not in valid_filter_operators:
                raise ValueError(f"Invalid filter operator: {operator}")

            if value is None:
                raise ValueError(f"Filter value is missing for column: {column}")

        # =========================================================
        # AGGREGATE VALIDATION
        # =========================================================

        if query_plan["operation"] == "AGGREGATE":
            group_by = query_plan.get("group_by", [])
            aggregations = query_plan.get("aggregations", [])

            if not isinstance(group_by, list):
                raise ValueError("AGGREGATE query requires group_by to be a list.")

            if not isinstance(aggregations, list):
                raise ValueError("AGGREGATE query requires aggregations to be a list.")

            for column in group_by:
                if column not in valid_columns:
                    raise ValueError(f"Invalid group_by column: {column}")

            for aggregation in aggregations:
                if not isinstance(aggregation, dict):
                    raise ValueError("Each aggregation must be an object.")

                column = aggregation.get("column")
                function = str(aggregation.get("function", "")).strip().lower()
                alias = aggregation.get("alias")

                if column not in valid_columns:
                    raise ValueError(f"Invalid aggregation column: {column}")

                if function not in valid_aggregation_functions:
                    raise ValueError(f"Invalid aggregation function: {function}")

                if not alias:
                    raise ValueError(
                        f"Aggregation alias is missing for column: {column}"
                    )

                aggregation["function"] = function

        # =========================================================
        # COUNT VALIDATION
        # =========================================================

        if query_plan["operation"] == "COUNT":
            column = query_plan.get("column")

            if column not in valid_columns:
                raise ValueError(f"Invalid COUNT column: {column}")

            if "distinct" not in query_plan:
                query_plan["distinct"] = False

        # =========================================================
        # SELECT / DISTINCT VALIDATION
        # =========================================================

        if query_plan["operation"] in {
            "SELECT",
            "DISTINCT",
            "FILTER_SELECT",
        }:
            select_columns = query_plan.get("select", [])

            if not isinstance(select_columns, list):
                raise ValueError("select must be a list.")

            for column in select_columns:
                if column not in valid_columns:
                    raise ValueError(f"Invalid select column: {column}")

        # =========================================================
        # SUM / AVERAGE / MIN / MAX VALIDATION
        # =========================================================

        if query_plan["operation"] in {
            "SUM",
            "AVERAGE",
            "MIN",
            "MAX",
            "SORT",
        }:
            column = query_plan.get("column")

            if column not in valid_columns:
                raise ValueError(f"Invalid {query_plan['operation']} column: {column}")

        # =========================================================
        # TOP_N VALIDATION
        # =========================================================

        if query_plan["operation"] == "TOP_N":
            column = query_plan.get("column")

            if column not in valid_columns:
                raise ValueError(f"Invalid TOP_N column: {column}")

            limit = query_plan.get("limit")

            if not isinstance(limit, int) or limit <= 0:
                raise ValueError("TOP_N limit must be a positive integer.")

            order = str(query_plan.get("order", "descending")).strip().lower()

            if order not in {"ascending", "descending"}:
                raise ValueError("TOP_N order must be ascending or descending.")

            query_plan["order"] = order

        # =========================================================
        # FINAL SAFETY CHECK
        # =========================================================

        serialized_plan = json.dumps(
            query_plan,
            ensure_ascii=False,
            indent=4,
        )

        if "count(*)" in serialized_plan.lower():
            raise ValueError("Invalid query plan: count(*) is not allowed.")

        if query_plan.get("operation") == "GROUP_BY":
            raise ValueError(
                "Invalid query plan: GROUP_BY must be normalized to AGGREGATE."
            )

        return serialized_plan

    def route_query_1(self, user_query, rag_context=""):
        """
        Route a user query to one of:

            STRUCTURED
            SEMANTIC
            HYBRID
            GENERAL

        The router only determines intent.
        It does not execute Pandas or ChromaDB queries.
        """

        valid_columns = {
            "risk_id",
            "project_id",
            "project_name",
            "date_reported",
            "risk_type",
            "description",
            "severity_level",
            "likelihood",
            "impact_days",
            "financial_impact_usd",
            "status",
            "mitigation_plan",
            "reported_by",
            "region",
            "tags",
        }

        prompt = f"""
    You are an Intent Router for a Construction Project Risk
    Analysis system.

    Your job is ONLY to determine how the user's question
    should be processed.

    You MUST return ONLY valid JSON.

    Do NOT answer the question.
    Do NOT generate Python.
    Do NOT generate SQL.
    Do NOT perform calculations.
    Do NOT retrieve data.

    =========================================================
    USER QUESTION
    =========================================================

    {user_query}

    =========================================================
    AVAILABLE DATA
    =========================================================

    The system contains construction risk records with these
    columns:

    risk_id
    project_id
    project_name
    date_reported
    risk_type
    description
    severity_level
    likelihood
    impact_days
    financial_impact_usd
    status
    mitigation_plan
    reported_by
    region
    tags

    =========================================================
    ROUTING OPTIONS
    =========================================================

    1. STRUCTURED

    Use STRUCTURED when the question can be answered using
    exact structured fields, filtering, sorting, counting,
    aggregation, or numerical calculations.

    Examples:

    "How many risks are there?"

    "How many risks are there in each region?"

    "How many Critical risks are there?"

    "What is the total financial impact?"

    "Show the top 5 risks by financial impact."

    "Show projects in the North region."

    "How many risks are there by risk type?"

    =========================================================

    2. SEMANTIC

    Use SEMANTIC when the question requires semantic
    understanding of text such as:

    description
    mitigation_plan
    tags

    Examples:

    "What risks are related to foundation instability?"

    "Find risks mentioning subcontractor delays."

    "What mitigation strategies are used for flooding?"

    "Find risks similar to foundation settlement."

    The answer requires semantic retrieval from the risk
    knowledge base / ChromaDB.

    =========================================================

    3. HYBRID

    Use HYBRID when the question contains BOTH:

    A. Exact structured conditions

    AND

    B. Semantic meaning/search requirements.

    Example:

    "Find Critical risks in the North region related to
    foundation instability."

    Structured conditions:

    severity_level = Critical
    region = North

    Semantic condition:

    foundation instability

    Another example:

    "Find high financial impact risks in South related to
    flooding."

    Structured:

    region = South

    financial_impact_usd = high

    Semantic:

    flooding

    =========================================================

    4. GENERAL

    Use GENERAL when the question is general knowledge or
    conversation and does not require searching the risk
    dataset.

    Examples:

    "What is a construction risk?"

    "What is the difference between likelihood and severity?"

    "Explain risk mitigation."

    "Hello"

    =========================================================
    STRUCTURED CONDITIONS
    =========================================================

    When structured conditions are present, extract them.

    Use ONLY these operators:

    equals
    not_equals
    in
    not_in
    contains
    greater_than
    less_than
    greater_than_or_equal
    less_than_or_equal

    Every condition must contain:

    column
    operator
    value

    Example:

    "Critical risks in North"

    should contain:

    {{
        "column": "severity_level",
        "operator": "equals",
        "value": "Critical"
    }}

    and:

    {{
        "column": "region",
        "operator": "equals",
        "value": "North"
    }}

    =========================================================
    GROUPING
    =========================================================

    Grouping language includes:

    each
    every
    by
    per
    for each

    Examples:

    "How many risks are there in each region?"

    This is STRUCTURED.

    The grouping concept is:

    region

    Do NOT create:

    region = "each region"

    Instead:

    requires_aggregation = true

    =========================================================
    SEMANTIC QUESTION
    =========================================================

    For SEMANTIC or HYBRID queries, extract the semantic
    portion of the question.

    Example:

    "Find Critical risks in North related to foundation
    instability."

    Return:

    semantic_question = "foundation instability"

    Do NOT include:

    Critical
    North

    in semantic_question because those are structured
    conditions.

    =========================================================
    AGGREGATION
    =========================================================

    Set:

    requires_aggregation = true

    when the user asks for:

    count
    how many
    total
    sum
    average
    minimum
    maximum
    top N
    grouped results
    by
    each
    per
    for each

    Otherwise use:

    requires_aggregation = false

    =========================================================
    OUTPUT FORMAT
    =========================================================

    Return exactly this JSON structure:

    {{
        "intent": "STRUCTURED",
        "structured_query": true,
        "semantic_query": false,
        "requires_aggregation": false,
        "structured_conditions": [],
        "semantic_question": "",
        "reason": ""
    }}

    intent MUST be exactly one of:

    STRUCTURED
    SEMANTIC
    HYBRID
    GENERAL

    =========================================================
    EXAMPLE 1
    =========================================================

    Question:

    "How many risks are there in each region?"

    Return:

    {{
        "intent": "STRUCTURED",
        "structured_query": true,
        "semantic_query": false,
        "requires_aggregation": true,
        "structured_conditions": [],
        "semantic_question": "",
        "reason": "The question requires an exact grouped count by region."
    }}

    =========================================================
    EXAMPLE 2
    =========================================================

    Question:

    "What risks are related to foundation instability?"

    Return:

    {{
        "intent": "SEMANTIC",
        "structured_query": false,
        "semantic_query": true,
        "requires_aggregation": false,
        "structured_conditions": [],
        "semantic_question": "foundation instability",
        "reason": "The question requires semantic search over risk descriptions."
    }}

    =========================================================
    EXAMPLE 3
    =========================================================

    Question:

    "Find Critical risks in North related to foundation
    instability."

    Return:

    {{
        "intent": "HYBRID",
        "structured_query": true,
        "semantic_query": true,
        "requires_aggregation": false,
        "structured_conditions": [
            {{
                "column": "severity_level",
                "operator": "equals",
                "value": "Critical"
            }},
            {{
                "column": "region",
                "operator": "equals",
                "value": "North"
            }}
        ],
        "semantic_question": "foundation instability",
        "reason": "The query contains exact structured filters and a semantic search requirement."
    }}

    =========================================================
    EXAMPLE 4
    =========================================================

    Question:

    "What is the difference between likelihood and severity?"

    Return:

    {{
        "intent": "GENERAL",
        "structured_query": false,
        "semantic_query": false,
        "requires_aggregation": false,
        "structured_conditions": [],
        "semantic_question": "",
        "reason": "The question asks for a general explanation rather than dataset retrieval."
    }}

    =========================================================
    STRICT RULES
    =========================================================

    1. Return JSON only.

    2. Never return markdown.

    3. Never return Python.

    4. Never return SQL.

    5. Never answer the user's question.

    6. Use only valid DataFrame columns.

    7. Never create a structured condition for grouping words.

    8. "each region" means grouping, NOT:
    region = "each region"

    9. "by region" means grouping, NOT:
    region = "by region"

    10. "per region" means grouping, NOT:
        region = "per region"

    11. Use HYBRID only when both structured and semantic
        requirements exist.

    12. Use SEMANTIC when semantic text retrieval is required
        without structured conditions.

    13. Use STRUCTURED when exact DataFrame operations are
        sufficient.

    14. Use GENERAL when the question does not require the
        risk dataset.

    15. structured_conditions must be a list.

    16. semantic_question must be a string.

    17. Return valid JSON.

    =========================================================
    FINAL RESPONSE
    =========================================================

    Return ONLY the JSON object.
    """

        # =========================================================
        # CALL LLM
        # =========================================================

        response = ollama.chat(
            model="llama3.2", messages=[{"role": "user", "content": prompt}]
        )

        raw_content = response["message"]["content"].strip()

        # =========================================================
        # EXTRACT JSON
        # =========================================================

        raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content, flags=re.IGNORECASE)

        raw_content = re.sub(r"\s*```$", "", raw_content).strip()

        try:
            route = json.loads(raw_content)

        except json.JSONDecodeError:
            start = raw_content.find("{")
            end = raw_content.rfind("}")

            if start == -1 or end == -1:
                raise ValueError(
                    f"LLM router did not return valid JSON:\n{raw_content}"
                )

            try:
                route = json.loads(raw_content[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid router JSON:\n{raw_content}") from exc

        # =========================================================
        # VALIDATE ROUTER RESULT
        # =========================================================

        if not isinstance(route, dict):
            raise ValueError("Router response must be a JSON object.")

        intent = str(route.get("intent", "")).strip().upper()

        valid_intents = {"STRUCTURED", "SEMANTIC", "HYBRID", "GENERAL"}

        if intent not in valid_intents:
            raise ValueError(f"Invalid routing intent: {intent}")

        route["intent"] = intent

        # =========================================================
        # NORMALIZE FLAGS
        # =========================================================

        route["structured_query"] = intent in {"STRUCTURED", "HYBRID"}

        route["semantic_query"] = intent in {"SEMANTIC", "HYBRID"}

        # =========================================================
        # NORMALIZE STRUCTURED CONDITIONS
        # =========================================================

        conditions = route.get("structured_conditions", [])

        if conditions is None:
            conditions = []

        if not isinstance(conditions, list):
            raise ValueError("structured_conditions must be a list.")

        normalized_conditions = []

        grouping_values = {
            "each region",
            "every region",
            "by region",
            "per region",
            "for each region",
            "each risk type",
            "every risk type",
            "by risk type",
            "per risk type",
            "for each risk type",
            "each severity level",
            "every severity level",
            "by severity level",
            "per severity level",
            "for each severity level",
        }

        for condition in conditions:
            if not isinstance(condition, dict):
                continue

            column = condition.get("column")
            operator = condition.get("operator")
            value = condition.get("value")

            if column not in valid_columns:
                raise ValueError(f"Invalid structured column: {column}")

            if not operator:
                operator = "equals"

            operator = str(operator).strip().lower()

            valid_operators = {
                "equals",
                "not_equals",
                "in",
                "not_in",
                "contains",
                "greater_than",
                "less_than",
                "greater_than_or_equal",
                "less_than_or_equal",
            }

            if operator not in valid_operators:
                raise ValueError(f"Invalid structured operator: {operator}")

            value_text = str(value).strip().lower() if value is not None else ""

            # Never allow grouping phrases to become filters.
            if value_text in grouping_values:
                continue

            if value is None:
                raise ValueError(f"Missing value for structured condition: {column}")

            normalized_conditions.append(
                {"column": column, "operator": operator, "value": value}
            )

        route["structured_conditions"] = normalized_conditions

        # =========================================================
        # NORMALIZE SEMANTIC QUESTION
        # =========================================================

        semantic_question = route.get("semantic_question", "")

        if semantic_question is None:
            semantic_question = ""

        route["semantic_question"] = str(semantic_question).strip()

        # =========================================================
        # AGGREGATION DETECTION
        # =========================================================

        question_lower = user_query.lower()

        aggregation_keywords = [
            "how many",
            "count",
            "total",
            "sum",
            "average",
            "avg",
            "minimum",
            "maximum",
            "min",
            "max",
            "top ",
            "highest",
            "lowest",
            "each ",
            "every ",
            "by ",
            "per ",
            "for each ",
        ]

        route["requires_aggregation"] = any(
            keyword in question_lower for keyword in aggregation_keywords
        )

        # =========================================================
        # FINAL ROUTING SAFETY
        # =========================================================

        if intent == "GENERAL":
            route["structured_query"] = False
            route["semantic_query"] = False
            route["structured_conditions"] = []
            route["semantic_question"] = ""
            route["requires_aggregation"] = False

        if intent == "STRUCTURED":
            route["semantic_query"] = False
            route["semantic_question"] = ""

        if intent == "SEMANTIC":
            route["structured_query"] = False
            route["structured_conditions"] = []

        # =========================================================
        # RETURN NORMALIZED ROUTE
        # =========================================================

        return json.dumps(route, ensure_ascii=False, indent=4)

    def route_query(self, user_query):

        question = user_query.strip()

        if not question:
            return json.dumps(
                {
                    "intent": "GENERAL",
                    "structured_query": False,
                    "semantic_query": False,
                    "requires_aggregation": False,
                    "structured_conditions": [],
                    "semantic_question": "",
                    "reason": "Empty user query.",
                },
                indent=2,
            )

        question_lower = question.lower()

        # ==========================================================
        # 1. GENERAL / CASUAL QUERIES
        # ==========================================================

        casual_queries = {
            "hello",
            "hi",
            "hey",
            "hello!",
            "hi!",
            "hey!",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "how are you?",
            "thanks",
            "thank you",
            "bye",
            "goodbye",
        }

        if question_lower in casual_queries:
            return json.dumps(
                {
                    "intent": "GENERAL",
                    "structured_query": False,
                    "semantic_query": False,
                    "requires_aggregation": False,
                    "structured_conditions": [],
                    "semantic_question": "",
                    "reason": "Casual conversation or greeting.",
                },
                indent=2,
            )

        # ==========================================================
        # 2. SEMANTIC INDICATORS
        # ==========================================================

        semantic_indicators = [
            "related to",
            "related with",
            "associated with",
            "associated to",
            "caused by",
            "caused due to",
            "similar to",
            "similar risks",
            "similar problems",
            "similar issues",
            "similar situations",
            "connected to",
            "linked to",
            "involving",
            "involve",
            "could affect",
            "could impact",
            "may affect",
            "may impact",
            "problems with",
            "issues with",
            "concerns about",
            "risks concerning",
            "risks associated with",
            "risks involving",
            "risks caused by",
        ]

        has_semantic_indicator = any(
            phrase in question_lower for phrase in semantic_indicators
        )

        # ==========================================================
        # 3. STRUCTURED INDICATORS
        # ==========================================================

        structured_indicators = [
            # Regions
            "north region",
            "south region",
            "east region",
            "west region",
            # Severity
            "critical",
            "high severity",
            "medium severity",
            "low severity",
            # Fields
            "severity =",
            "region =",
            "status =",
            "risk type",
            # Numeric fields
            "financial impact",
            "impact days",
            # Numeric comparisons
            "greater than",
            "less than",
            "greater than or equal",
            "less than or equal",
            "above",
            "below",
            "more than",
            # Aggregation
            "how many",
            "count",
            "total",
            "sum",
            "average",
            "mean",
            "minimum",
            "maximum",
            "highest",
            "lowest",
            # Grouping
            "each region",
            "every region",
            "per region",
            "by region",
            "for each region",
            "each project",
            "every project",
            "per project",
            "by project",
            "group by",
            # Sorting / Top N
            "top ",
            "bottom ",
            "sort by",
            "sorted by",
            # Explicit filtering
            "where",
            "filter by",
        ]

        has_structured_indicator = any(
            phrase in question_lower for phrase in structured_indicators
        )

        # ==========================================================
        # 4. DETERMINE SEMANTIC VS HYBRID
        # ==========================================================

        # ----------------------------------------------------------
        # Semantic-only
        #
        # Example:
        # "Find risks related to foundation instability."
        # ----------------------------------------------------------

        if has_semantic_indicator and not has_structured_indicator:
            semantic_question = question

            # Remove generic search wording
            semantic_question = re.sub(
                r"^\s*find\s+risks?\s+(related to|associated with|"
                r"caused by|similar to|involving|connected to|linked to)\s+",
                "",
                semantic_question,
                flags=re.IGNORECASE,
            ).strip()

            return json.dumps(
                {
                    "intent": "SEMANTIC",
                    "structured_query": False,
                    "semantic_query": True,
                    "requires_aggregation": False,
                    "structured_conditions": [],
                    "semantic_question": semantic_question,
                    "reason": (
                        "The query requires meaning-based semantic "
                        "retrieval and does not contain an exact "
                        "structured condition."
                    ),
                },
                indent=2,
            )

        # ==========================================================
        # 5. HYBRID
        # ==========================================================

        if has_semantic_indicator and has_structured_indicator:
            return json.dumps(
                {
                    "intent": "HYBRID",
                    "structured_query": True,
                    "semantic_query": True,
                    "requires_aggregation": False,
                    "structured_conditions": [],
                    "semantic_question": question,
                    "reason": (
                        "The query contains both structured "
                        "conditions and semantic search requirements. "
                        "LLM will extract the exact structured "
                        "conditions."
                    ),
                },
                indent=2,
            )

        # ==========================================================
        # 6. STRUCTURED
        # ==========================================================

        if has_structured_indicator:
            return json.dumps(
                {
                    "intent": "STRUCTURED",
                    "structured_query": True,
                    "semantic_query": False,
                    "requires_aggregation": (
                        "how many" in question_lower
                        or "count" in question_lower
                        or "total" in question_lower
                        or "sum" in question_lower
                        or "average" in question_lower
                        or "each region" in question_lower
                        or "by region" in question_lower
                    ),
                    "structured_conditions": [],
                    "semantic_question": "",
                    "reason": (
                        "The query requires exact structured "
                        "data filtering or aggregation."
                    ),
                },
                indent=2,
            )

        # ==========================================================
        # 7. LLM FALLBACK
        # ==========================================================

        prompt = f"""
    You are the Intent Router for a Construction Project
    Risk Analyzer.

    Classify the user's question into exactly one:

    STRUCTURED
    SEMANTIC
    HYBRID
    GENERAL

    =========================================================
    VERY IMPORTANT
    =========================================================

    SEMANTIC means the user wants risks based on meaning,
    similarity, concept, description, or natural-language
    relationship.

    Examples:

    "Find risks related to foundation instability."

    "Find risks associated with weak soil conditions."

    "Find risks related to ground settlement."

    "Find risks involving structural instability."

    "Find risks caused by poor soil bearing capacity."

    "Find risks similar to foundation problems."

    These are SEMANTIC.

    Do NOT classify them as STRUCTURED simply because they
    start with "Find risks".

    ---------------------------------------------------------

    STRUCTURED means the user asks for exact data operations.

    Examples:

    "How many risks are there?"

    "How many risks are there in each region?"

    "Find Critical risks."

    "Find risks in the North region."

    "What is the total financial impact?"

    "Show the top 5 risks by financial impact."

    These are STRUCTURED.

    ---------------------------------------------------------

    HYBRID means BOTH exact structured conditions and semantic
    meaning are required.

    Example:

    "Find Critical risks in the North region related to
    foundation instability."

    Structured:
    severity = Critical
    region = North

    Semantic:
    foundation instability

    ---------------------------------------------------------

    GENERAL means casual conversation or questions outside
    the Construction Project Risk Analyzer.

    Example:

    "Hello"

    "How are you?"

    "What is Python?"

    =========================================================
    USER QUESTION
    =========================================================

    {question}

    =========================================================

    Return ONLY JSON:

    {{
        "intent": "STRUCTURED | SEMANTIC | HYBRID | GENERAL",
        "structured_query": true,
        "semantic_query": false,
        "requires_aggregation": false,
        "structured_conditions": [],
        "semantic_question": "",
        "reason": ""
    }}
    """

        try:
            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict query intent classification router."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0},
            )

            content = response["message"]["content"].strip()

            # ------------------------------------------------------
            # Extract JSON
            # ------------------------------------------------------

            start = content.find("{")
            end = content.rfind("}")

            if start >= 0 and end > start:
                content = content[start : end + 1]

            plan = json.loads(content)

            # ------------------------------------------------------
            # Normalize
            # ------------------------------------------------------

            intent = str(plan.get("intent", "GENERAL")).upper().strip()

            if intent not in {"STRUCTURED", "SEMANTIC", "HYBRID", "GENERAL"}:
                intent = "GENERAL"

            plan["intent"] = intent

            return json.dumps(plan, indent=2)

        except Exception as e:
            return json.dumps(
                {
                    "intent": "GENERAL",
                    "structured_query": False,
                    "semantic_query": False,
                    "requires_aggregation": False,
                    "structured_conditions": [],
                    "semantic_question": "",
                    "reason": (
                        "Unable to confidently classify the "
                        "question. Router error: "
                        f"{str(e)}"
                    ),
                },
                indent=2,
            )

    # ============================================================
    # HYBRID QUERY
    # ============================================================

    def execute_hybrid_query(self, route_plan, csv_file_path, top_k=10):
        """
        Execute a hybrid construction risk query.

        Hybrid query contains:
            1. Structured conditions
            2. Semantic question

        Example:

            User:
            "Find Critical risks in the North region
             related to foundation instability."

        Structured:
            severity_level = Critical
            region = North

        Semantic:
            foundation instability

        Processing:

            Pandas
                ↓
            Structured candidate risk IDs

            ChromaDB
                ↓
            Semantic candidate risk IDs

            Intersection
                ↓
            Final risk records

            LLM
                ↓
            Final natural-language answer
        """

        try:
            # ====================================================
            # 1. Validate route plan
            # ====================================================

            if not isinstance(route_plan, dict):
                return {
                    "success": False,
                    "message": "route_plan must be a dictionary.",
                    "data": [],
                    "answer": "",
                }

            intent = str(route_plan.get("intent", "")).upper().strip()

            if intent != "HYBRID":
                return {
                    "success": False,
                    "message": (f"Invalid intent '{intent}'. Expected HYBRID."),
                    "data": [],
                    "answer": "",
                }

            structured_conditions = route_plan.get("structured_conditions", [])

            semantic_question = str(route_plan.get("semantic_question", "")).strip()

            original_question = str(route_plan.get("original_question", "")).strip()

            # ====================================================
            # 2. Validate semantic question
            # ====================================================

            if not semantic_question:
                return {
                    "success": False,
                    "message": ("Hybrid query requires semantic_question."),
                    "data": [],
                    "answer": "",
                }

            # ====================================================
            # 3. Load CSV
            # ====================================================

            df = self.data_load(csv_file_path)

            if df is None:
                return {
                    "success": False,
                    "message": "Unable to load risk dataset.",
                    "data": [],
                    "answer": "",
                }

            if df.empty:
                return {
                    "success": False,
                    "message": "Risk dataset is empty.",
                    "data": [],
                    "answer": "",
                }

            # ====================================================
            # 4. Validate risk_id column
            # ====================================================

            if "risk_id" not in df.columns:
                return {
                    "success": False,
                    "message": ("Dataset does not contain 'risk_id' column."),
                    "data": [],
                    "answer": "",
                }

            # ====================================================
            # 5. Apply STRUCTURED conditions using Pandas
            # ====================================================

            structured_df = self.apply_hybrid_conditions(df, structured_conditions)

            # ====================================================
            # 6. Get structured candidate IDs
            # ====================================================

            structured_ids = set(
                structured_df["risk_id"].dropna().astype(str).str.strip()
            )

            # ====================================================
            # 7. Execute SEMANTIC search using ChromaDB
            # ====================================================

            semantic_results = self.query_chroma_db(semantic_question, top_k=top_k)

            # ====================================================
            # 8. Extract semantic risk IDs
            # ====================================================

            semantic_ids = self.extract_semantic_risk_ids(semantic_results)

            # ====================================================
            # 9. INTERSECTION
            # ====================================================

            final_ids = structured_ids.intersection(semantic_ids)

            # ====================================================
            # 10. Get final records from original dataframe
            # ====================================================

            if final_ids:
                final_df = df[
                    df["risk_id"].astype(str).str.strip().isin(final_ids)
                ].copy()

            else:
                final_df = pd.DataFrame(columns=df.columns)

            # ====================================================
            # 11. Convert final records to dictionaries
            # ====================================================

            final_records = final_df.to_dict(orient="records")

            # ====================================================
            # 12. No matching records
            # ====================================================

            if not final_records:
                return {
                    "success": True,
                    "message": (
                        "No risks matched both the "
                        "structured conditions and "
                        "semantic query."
                    ),
                    "structured_count": len(structured_ids),
                    "semantic_count": len(semantic_ids),
                    "final_count": 0,
                    "structured_conditions": structured_conditions,
                    "semantic_question": semantic_question,
                    "data": [],
                    "answer": (
                        "No construction risks matched "
                        "both the specified conditions "
                        "and semantic criteria."
                    ),
                }

            # ====================================================
            # 13. Generate final LLM answer
            # ====================================================

            answer = self.generate_hybrid_answer(
                user_question=original_question,
                structured_conditions=structured_conditions,
                semantic_question=semantic_question,
                evidence=final_records,
            )

            # ====================================================
            # 14. Return complete result
            # ====================================================

            return {
                "success": True,
                "structured_count": len(structured_ids),
                "semantic_count": len(semantic_ids),
                "final_count": len(final_records),
                "structured_conditions": structured_conditions,
                "semantic_question": semantic_question,
                "data": final_records,
                "answer": answer,
            }

        except Exception as e:
            return {
                "success": False,
                "message": (f"Hybrid query failed: {str(e)}"),
                "data": [],
                "answer": "",
            }

    # ============================================================
    # APPLY STRUCTURED CONDITIONS
    # ============================================================

    def apply_hybrid_conditions(self, df, conditions):
        """
        Apply structured conditions using Pandas.

        Supported operators:

            equals
            not_equals
            contains
            starts_with
            ends_with
            greater_than
            greater_than_or_equal
            less_than
            less_than_or_equal
            in
            not_in
        """

        result_df = df.copy()

        if not conditions:
            return result_df

        for condition in conditions:
            if not isinstance(condition, dict):
                continue

            column = condition.get("column")

            operator = condition.get("operator", "equals")

            value = condition.get("value")

            # ----------------------------------------------------
            # Validate column
            # ----------------------------------------------------

            if not column:
                continue

            if column not in result_df.columns:
                continue

            # ----------------------------------------------------
            # Normalize operator
            # ----------------------------------------------------

            operator = str(operator).strip().lower()

            # ----------------------------------------------------
            # Ignore invalid grouping phrases
            # ----------------------------------------------------

            if isinstance(value, str):
                grouping_phrases = {
                    "each region",
                    "every region",
                    "per region",
                    "by region",
                    "for each region",
                    "each project",
                    "every project",
                    "per project",
                    "by project",
                    "for each project",
                }

                if value.lower().strip() in grouping_phrases:
                    continue

            # ----------------------------------------------------
            # EQUALS
            # ----------------------------------------------------

            if operator == "equals":
                result_df = result_df[
                    result_df[column].astype(str).str.strip().str.lower()
                    == str(value).strip().lower()
                ]

            # ----------------------------------------------------
            # NOT EQUALS
            # ----------------------------------------------------

            elif operator == "not_equals":
                result_df = result_df[
                    result_df[column].astype(str).str.strip().str.lower()
                    != str(value).strip().lower()
                ]

            # ----------------------------------------------------
            # CONTAINS
            # ----------------------------------------------------

            elif operator == "contains":
                result_df = result_df[
                    result_df[column]
                    .astype(str)
                    .str.contains(str(value), case=False, na=False)
                ]

            # ----------------------------------------------------
            # STARTS WITH
            # ----------------------------------------------------

            elif operator == "starts_with":
                result_df = result_df[
                    result_df[column].astype(str).str.startswith(str(value), na=False)
                ]

            # ----------------------------------------------------
            # ENDS WITH
            # ----------------------------------------------------

            elif operator == "ends_with":
                result_df = result_df[
                    result_df[column].astype(str).str.endswith(str(value), na=False)
                ]

            # ----------------------------------------------------
            # GREATER THAN
            # ----------------------------------------------------

            elif operator == "greater_than":
                numeric_values = pd.to_numeric(result_df[column], errors="coerce")

                result_df = result_df[numeric_values > float(value)]

            # ----------------------------------------------------
            # GREATER THAN OR EQUAL
            # ----------------------------------------------------

            elif operator == "greater_than_or_equal":
                numeric_values = pd.to_numeric(result_df[column], errors="coerce")

                result_df = result_df[numeric_values >= float(value)]

            # ----------------------------------------------------
            # LESS THAN
            # ----------------------------------------------------

            elif operator == "less_than":
                numeric_values = pd.to_numeric(result_df[column], errors="coerce")

                result_df = result_df[numeric_values < float(value)]

            # ----------------------------------------------------
            # LESS THAN OR EQUAL
            # ----------------------------------------------------

            elif operator == "less_than_or_equal":
                numeric_values = pd.to_numeric(result_df[column], errors="coerce")

                result_df = result_df[numeric_values <= float(value)]

            # ----------------------------------------------------
            # IN
            # ----------------------------------------------------

            elif operator == "in":
                if not isinstance(value, list):
                    value = [value]

                allowed_values = {str(v).strip().lower() for v in value}

                result_df = result_df[
                    result_df[column]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(allowed_values)
                ]

            # ----------------------------------------------------
            # NOT IN
            # ----------------------------------------------------

            elif operator == "not_in":
                if not isinstance(value, list):
                    value = [value]

                excluded_values = {str(v).strip().lower() for v in value}

                result_df = result_df[
                    ~result_df[column]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(excluded_values)
                ]

        return result_df

    # ============================================================
    # EXTRACT SEMANTIC RISK IDs
    # ============================================================

    def extract_semantic_risk_ids(self, semantic_results):
        """
        Extract risk_id values from ChromaDB results.

        Supports common ChromaDB response formats.
        """

        semantic_ids = set()

        if not semantic_results:
            return semantic_ids

        # ========================================================
        # FORMAT 1
        #
        # [
        #   {
        #       "metadata": {
        #           "risk_id": "R001"
        #       }
        #   }
        # ]
        # ========================================================

        if isinstance(semantic_results, list):
            for result in semantic_results:
                if not isinstance(result, dict):
                    continue

                metadata = result.get("metadata", {})

                if isinstance(metadata, dict):
                    risk_id = metadata.get("risk_id")

                    if risk_id is not None:
                        semantic_ids.add(str(risk_id).strip())

        # ========================================================
        # FORMAT 2
        #
        # ChromaDB native:
        #
        # {
        #   "ids": [["R001", "R002"]],
        #   "metadatas": [[...]]
        # }
        # ========================================================

        elif isinstance(semantic_results, dict):
            metadatas = semantic_results.get("metadatas")

            ids = semantic_results.get("ids")

            # ----------------------------------------------------
            # Metadata-based extraction
            # ----------------------------------------------------

            if metadatas:
                for metadata_group in metadatas:
                    if not metadata_group:
                        continue

                    for metadata in metadata_group:
                        if not isinstance(metadata, dict):
                            continue

                        risk_id = metadata.get("risk_id")

                        if risk_id is not None:
                            semantic_ids.add(str(risk_id).strip())

            # ----------------------------------------------------
            # IDs-based extraction
            # ----------------------------------------------------

            if ids:
                for id_group in ids:
                    if not id_group:
                        continue

                    for risk_id in id_group:
                        if risk_id is not None:
                            semantic_ids.add(str(risk_id).strip())

        return semantic_ids

    # ============================================================
    # GENERATE HYBRID ANSWER
    # ============================================================

    def generate_hybrid_answer(
        self, user_question, structured_conditions, semantic_question, evidence
    ):
        """
        Generate the final answer using Llama 3.2.

        IMPORTANT:

        The LLM receives ONLY the final intersected
        evidence.

        It does NOT perform the filtering itself.
        """

        try:
            # ====================================================
            # Prepare evidence
            # ====================================================

            evidence_json = json.dumps(evidence, indent=2, default=str)

            conditions_json = json.dumps(structured_conditions, indent=2, default=str)

            # ====================================================
            # Prompt
            # ====================================================

            prompt = f"""
You are a Construction Project Risk Analysis Assistant.

You must answer the user's question using ONLY the
construction risk records provided below.

============================================================
USER QUESTION
============================================================

{user_question}

============================================================
STRUCTURED CONDITIONS
============================================================

{conditions_json}

============================================================
SEMANTIC QUESTION
============================================================

{semantic_question}

============================================================
FINAL MATCHED RISK RECORDS
============================================================

{evidence_json}

============================================================
INSTRUCTIONS
============================================================

1. Use ONLY the supplied final matched risk records.

2. Do NOT invent information.

3. Do NOT assume information that is not present
   in the records.

4. Explain the result in a clear and concise manner.

5. For each important risk, mention relevant information
   such as:

   - risk_id
   - project_name
   - risk_type
   - severity_level
   - likelihood
   - impact_days
   - financial_impact_usd
   - description
   - mitigation_plan
   - status
   - region

6. If the user asks about a specific risk characteristic,
   focus on that characteristic.

7. If multiple risks are returned, summarize the common
   pattern as well as important differences.

8. Do not perform unsupported calculations.

9. Do not claim that a risk exists if it is not present
   in the supplied evidence.

10. If the evidence is empty, clearly state that no
    matching risks were found.

11. Keep the response suitable for a Construction Risk
    Analysis application.

============================================================
ANSWER
============================================================
"""

            # ====================================================
            # LLM CALL
            # ====================================================

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a construction project "
                            "risk analysis assistant. "
                            "Use only supplied evidence."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            # ====================================================
            # Extract response
            # ====================================================

            if not response:
                return "Unable to generate the final answer."

            message = response.get("message", {})

            answer = message.get("content", "")

            if not answer:
                return "Unable to generate the final answer."

            return answer.strip()

        except Exception as e:
            return f"Unable to generate the hybrid answer. Error: {str(e)}"
