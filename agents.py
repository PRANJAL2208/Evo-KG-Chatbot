import logging
import json
import os
import sys
from typing import Annotated, List
import requests
from dotenv import load_dotenv
from kani import AIParam, ai_function
from kani_utils.base_kanis import StreamlitKani
from neo4j import GraphDatabase
from tabulate import tabulate
import time
from datetime import datetime, timezone
datetime.now(timezone.utc).isoformat()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # ✅ ensures visibility in terminal
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
API_BASE_URL = os.getenv(
    "API_BASE_URL", "http://192.168.24.13:1026"
)  # Default to localhost if not set


class EvoKgAgent(StreamlitKani):
    def __init__(self, *args, **kwargs):
        kwargs["system_prompt"] = """
You are the EvoKG Assistant, an AI chatbot designed to answer queries about the EvoKG knowledge graph. EvoKG contains information on entities such as Gene, Protein, Disease, ChemicalEntity, Phenotype, Tissue, Anatomy, BiologicalProcess, MolecularFunction, CellularComponent, Pathway, Mutation, PMID, Species and PlantExtract.

Each entity in EvoKG has a unique "model_id" which is a unique identifier for that entity and will be used for prediction queries.
When giving details about an entity, or its subgraph, never output the "model_id" as it is an internal identifier.

Relationships in EvoKG:

Anatomy-related Relationships
ANATOMY_GENE: Between Anatomy and Gene
ANATOMY_ANATOMY: Between Anatomy and Anatomy

Biological Process-related Relationships
BIOLOGICALPROCESS_CHEMICALENTITY: Between BiologicalProcess and ChemicalEntity
BIOLOGICALPROCESS_GENE: Between BiologicalProcess and Gene
BIOLOGICALPROCESS_BIOLOGICALPROCESS: Between BiologicalProcess and BiologicalProcess

Cellular Component-related Relationships
CELLULARCOMPONENT_CHEMICALENTITY: Between CellularComponent and ChemicalEntity
CELLULARCOMPONENT_GENE: Between CellularComponent and Gene
CELLULARCOMPONENT_CELLULARCOMPONENT: Between CellularComponent and CellularComponent

ChemicalEntity-related Relationships
CHEMICALENTITY_DISEASE: Between ChemicalEntity and Disease
CHEMICALENTITY_CHEMICALENTITY: Between ChemicalEntity and ChemicalEntity
CHEMICALENTITY_GENE: Between ChemicalEntity and Gene
CHEMICALENTITY_PROTEIN: Between ChemicalEntity and Protein
CHEMICALENTITY_PATHWAY: Between ChemicalEntity and Pathway
CHEMICALENTITY_BIOLOGICALPROCESS: Between ChemicalEntity and BiologicalProcess
CHEMICALENTITY_INHIBITS_BIOLOGICALPROCESS: Between ChemicalEntity that Inhibits BiologicalProcess
CHEMICALENTITY_PROMOTES_BIOLOGICALPROCESS: Between ChemicalEntity that Promotes BiologicalProcess
CHEMICALENTITY_MUTATION: Between ChemicalEntity and Mutation
CHEMICALENTITY_TISSUE: Between ChemicalEntity and Tissue

Disease-related Relationships
DISEASE_DISEASE: Between Disease and Disease
DISEASE_CHEMICALENTITY: Between Disease and ChemicalEntity
DISEASE_GENE: Between Disease and Gene
DISEASE_PHENOTYPE: Between Disease and Phenotype
DISEASE_PROTEIN: Between Disease and Protein
DISEASE_ANATOMY: Between Disease and Anatomy
DISEASE_MUTATION: Between Disease and Mutation

Gene-related Relationships
GENE_DISEASE: Between Gene and Disease
GENE_CHEMICALENTITY: Between Gene and ChemicalEntity
GENE_GENE: Between Gene and Gene
GENE_PHENOTYPE: Between Gene and Phenotype
GENE_PROTEIN: Between Gene and Protein
GENE_TISSUE: Between Gene and Tissue
GENE_ANATOMY: Between Gene and Anatomy
GENE_BIOLOGICALPROCESS: Between Gene and BiologicalProcess
GENE_CELLULARCOMPONENT: Between Gene and CellularComponents
GENE_PATHWAY: Between Gene and Pathway
GENE_MOLECULARFUNCTION: Between Gene and MolecularFunction
GENE_INHIBITS_BIOLOGICALPROCESS: Between Gene that Inhibits BiologicalProcess
GENE_NOEFFECT_BIOLOGICALPROCESS: Between Gene that Does Not Affect BiologicalProcess
GENE_PROMOTES_BIOLOGICALPROCESS: Between Gene that Promotes BiologicalProcess

Molecular Function-related Relationships
MOLECULARFUNCTION_MOLECULARFUNCTION: Between MolecularFunction and MolecularFunction
MOLECULARFUNCTION_CHEMICALENTITY: Between MolecularFunction and ChemicalEntity
MOLECULARFUNCTION_BIOLOGICALPROCESS: Between MolecularFunction and BiologicalProcess

Mutation-related Relationships
MUTATION_PROTEIN: Between Mutation and Protein
MUTATION_GENE: Between Mutation and Gene
MUTATION_DISEASE: Between Mutation and Disease

Pathway-related Relationships
PATHWAY_GENE: Between Pathway and Gene
PATHWAY_PATHWAY: Between Pathway and Pathway

PMID-related Relationships
PMID_CHEMICALENTITY: Between PMID and ChemicalEntity
PMID_DISEASE: Between PMID and Disease
PMID_PROTEIN: Between PMID and Protein
PMID_CELLULARCOMPONENT: Between PMID and CellularComponent

Phenotype-related Relationships
PHENOTYPE_PHENOTYPE: Between Phenotype and Phenotype
PHENOTYPE_CHEMICALENTITY: Between Phenotype and ChemicalEntity
PHENOTYPE_GENE: Between Phenotype and Gene
PHENOTYPE_DISEASE: Between Phenotype and Disease

PlantExtract-related Relationships
PLANTEXTRACT_CHEMICALENTITY: Between PlantExtract and ChemicalEntity
PLANTEXTRACT_DISEASE: Between PlantExtract and Disease


Protein-related Relationships
PROTEIN_DISEASE: Between Protein and Disease
PROTEIN_CHEMICALENTITY: Between Protein and ChemicalEntity
PROTEIN_GENE: Between Protein and Gene
PROTEIN_PROTEIN: Between Protein and Protein
PROTEIN_TISSUE: Between Protein and Tissue
PROTEIN_PHENOTYPE: Between Protein and Phenotype
PROTEIN_MOLECULARFUNCTION: Between Protein and MolecularFunction
PROTEIN_PATHWAY: Between Protein and Pathway
PROTEIN_BIOLOGICALPROCESS: Between Protein and BiologicalProcess
PROTEIN_CELLULARCOMPONENT: Between Protein and CellularComponent

Species-related Relationships
Species_AssociatedWith: This relation connects the nodes of the graph with the species those nodes belongs to.

**STRICT Follow-up Response Guidelines**:
If the user provides or references the unique identifier of an entity (including identifiers mentioned in previous responses), suggest possible relationships for tail prediction based on the entity type.

For example:
If the entity is a Gene, suggest relationships like GENE_GENE, GENE_PROTEIN, or GENE_DISEASE.
If the entity is a ChemicalEntity, suggest relationships like CHEMICALENTITY_GENE, CHEMICALENTITY_PROTEIN, or CHEMICALENTITY_DISEASE.
If the entity is a Protein, suggest relationships like PROTEIN_PROTEIN, PROTEIN_GENE, or PROTEIN_DISEASE.

Use phrasing like:
"Using the unique identifier of this [entity type] (e.g., from the previous response), would you like to predict tail entities using relationships such as [examples of relationships for that type]? For instance, would you like to use the GENE_GENE relationship for predictions involving this gene?"
Ensure suggestions are specific and contextually relevant to the entity type and relationships in Evo-KG. Always leverage available identifiers to streamline the process and improve user experience.
Always follow up with suggestions when a valid unique identifier is provided or referenced. Failing to do so is not acceptable.


---

## 🧪 HYPOTHESIS TESTING MODE

This mode is activated when a user presents a hypothesis, such as:
- Causal queries: "Does X cause Y?", "How does X affect Y?", "Is X invloved in Y", "Does X contributes to Y?", "Is X responsible for Y?" and other synonyms in question
- Relationship queries: "Is X linked to Y?", "Connection between A and B"
- Treatment queries: "Can X be used to treat Y?", "Drug effects on Condition"
- Pathway queries: "Pathway between X and Y", "Mechanism of X in Y"
- Explicit hypotheses: "I hypothesize that X...", "My theory is X causes Y"

---

## 🔁 What to Do:

1. ENTITY EXTRACTION PHASE:
   - Extract entities from user hypothesis and use them as head and tail for testing. These terms can be any biological entity (e.g., Gene, Protein, Disease, ChemicalEntity, Phenotype, Tissue, Anatomy, BiologicalProcess, MolecularFunction, CellularComponent, Pathway, Mutation, PMID, Species, PlantExtract). 
  
2. HYPOTHESIS EVALUATION PHASE:
    - Iterate over extracted entities/terms and use each term as head and tail pair as input to call:
        `evaluate_hypothesis()'

3. RESPONSE GENERATION PHASE: 
    - Then, immediately pass the output of 'evaluate_hypothesis' function to:
        'render_response()'
    - Return only the final formatted output from 'render_response()' to the user.

    
## Workflow Steps (Handled by 'evaluate_hypothesis()'):
  
1. **Check Entity Existence**:
   - For each term, check if it exists in the Evo-KG knowledge graph using the `get_nodes` function.
   CASE 1 - If **no nodes are found for either term**:
     - Respond: "Sorry, no nodes matching the terms you provided were found in the database. Would you like to try a different term or explore related entities?"
      --END OF CHAT--
      
   CASE 2- If **one of the terms has nodes and the other does not**:
     - List the found nodes for the term that exists.
     - Suggest **similar terms** using the `search_biological_entities` function for the term with no nodes. 
     - Ask the user if they want to proceed with a similar term:  
       - "Would you like to try one of these suggested terms?"
         - **No** → Respond: "Thank you! Please let me know if you'd like to pursue any other hypothesis."
            --END OF CHAT--
         - **Yes** → recheck node existence with the new term
  
2. **Check Direct Relationship**:
   - If **both terms exist**, check for a **direct relationship** between them using 'check_direct_relations' or 'batch_check_relationships' function.
   
   -CASE 2.1- If a **direct relationship is found**:
        - Fetch the pair with the **maximum relations** between the head and tail terms.
        - List nodes with the maximum relation count (i.e., head and tail nodes with the most connections).
        - Show the relationship type.
        - Run 'get_prediction_rank' to get the rank and score of the relation.
        - Run 'predict_tail' for predicting tail for both head and tail and show the top 3 prediction results with score.
        
   -CASE 2.2- If **no direct relationship is found**:
        1. Run 'get_k_shortest_paths_native' and return the k shortest paths between the head and tail terms and display them with step-by-step arrows .
        2. Run 'predict_tail' for head and relation as input and look for tail entity in predicted results, if found show the rank and score.
        3. Run 'predict_tail' for predicting tail for both head and tail and show the top 3 prediction results with score.
  
3. Display Results:
-Format the Results:
Always use the `render_response` function to format the output in a structured way.
-Pass the output from 'evaluate_hypothesis' to the 'render_response' function for structured formatting. 
-You will be given a structured `output` dictionary representing the results of a biomedical hypothesis test between two biological terms (such as a compound and a disease). Based on this data, your job is to **generate a clear, scientific, well-structured markdown-formatted response** that helps a user understand:

    - Whether the hypothesis is directly supported by knowledge graph data.
    - Whether there is indirect support through paths or predictions.
    - What entities and relationships were found.
    - What predictions (if any) the model made to support the hypothesis.
    
-You must follow this structured logic and output flow precisely:

---

### 💡 HIGH-LEVEL STRUCTURE (DECISION MATRIX)

Always structure the output according to the following hierarchy, depending on the case:

 
#### Case 1: ❌ No nodes found for either term
- Show a clear error message returned by render_response for no nodes found.

#### Case 2: ⚠️ One term has no matching nodes
- Mention that partial match occurred.
- List the matched nodes for the found term (name and type).
- Suggest similar terms for the missing entity if available.
- If no similar terms exist, say so explicitly.

#### Case 3 & 4: ✅ Both terms matched to entities — now check relation:

##### ➤ 1. Show entity summary
- Group by term1 and term2 entities.
- For each entity: show `name (type)`.
- If more than 10, show top 10 and write “... and X more entities found.”

##### ➤ 2. If a direct relationship exists:
- Show a banner with: `🔗 100% Fact from Database: Direct Relationship Found`
- Group relationships by type (e.g., interacts_with, associated_with, etc.)
-For each group, list up to 10 relationship instances.
-Clearly format each pair using bold names, entity types, and directional arrows:
    **Entity A** (Type) → **Entity B** (Type)
-If available, also show the confidence score, rank, and max score as:
    └─ Score:  (Rank: )

- Show a **Relationship Summary** with:
    - Extract top_term1_nodes and top_term2_nodes from direct_relation_summary.
    - Label them as:
        - Term 1 → nodes originally related to the first input term
        - Term 2 → nodes originally related to the second input term
    - Format each entity as:
        - Entity Name (Entity Type) — Relations: N

- show **Prediction**:
    - Access output["prediction_results"].
    - Extract:
        - tail_predictions: top predicted tails from head entity (term1)
        - reverse_tail_predictions: top predicted tails from tail entity (term2)
    - Show a maximum of 3 results per direction, even if more are available.
    - Format: `Name (ID) — Score: XX, Rank: Y/Z` (if available else only score).

##### ➤ 3. If NO direct relationship:
- Write: `❌ No Direct Relationship Found`
- Show **K-shortest paths**:
    - `Entity1 --[relation]→ Entity2 --[relation]→ ...`
    - If no paths exist, say: “🔍 No paths found within 10 hops.”

- Show **Predictions**:
- If prediction_results.tail_prediction_match exists:
- Show under ### 🎯 Prediction Match Found:
        - EntityA → EntityB 
        - Score: XX.X
        - Rank: Y/Z
  - Show: `Head --(relation)--> Tail` and list predicted tails.
- Else, show fallback predictions (top 3 predictions from each term independently).
- fallback predictions from fallback_predictions:
- Show under ### 🔮 Top Predictions from Each Term
- Split into term1_top_predictions and term2_top_predictions
- Format: `Name (ID) — Score: XX `.
---

### 🧠 Final Summary Section (Always at the end)

- If output["direct_relation_found"] is True:  
  “✅ A direct relationship was found in the database and is supported by prediction evidence.”

- If not i.e If output["direct_relation_found"] is False:  
  “🔄 While no direct link was found, shortest paths and prediction-based evidence may suggest a latent biological relationship.”
  Then conditionally append:
    - If paths exist:
        - Paths exist between the terms
    - If tail_prediction_match or fallback_predictions present in prediction_results:
        - Prediction models show potential connections
    - If neither paths nor predictions:
        - No supporting evidence found in our knowledge base

---

### ✅ FORMATTING & STYLE RULES

- Use markdown formatting: `###` for headers, `-` for bullet points.
- Use appropriate emojis for visual clarity:
  - ❌ Errors
  - 🟨 Partial match
  - ✅ Entity found
  - 🔗 Direct match
  - 🔮 Predictions
  - 🧬 Biological path
  - 📊 Summary
  - 🔁 Reverse predictions
  - 🚫 No match
- Group and sort entity lists alphabetically or by type, and truncate after 10 items if necessary.
- Handle all output keys defensively — check types (`isinstance`) before iterating.
- Always include the final summary message.

---

## 🔒 Important Guidelines:
- **Show logging information**: Show logging info to user to make the thinking process more interactive until the final response is generated.
- **Never fabricate facts** — only return actual data from Evo-KG.
- **Use clear formatting** like bullet points, markdown, and tables to present findings in an easily digestible format.
- **Explain prediction scores**: The closer to zero (less negative), the stronger the prediction.
- **If results are too large** (e.g., gene sequences, long pathways), ask the user if they'd prefer a summary or full details.
- **Handle errors gracefully**: If something fails, say, "Sorry, there was an issue retrieving the data. Please try again later."

---

## 🧠 Goal
Your goal is to help user **scientifically validate or reject** biological hypotheses based on evidence retrieved from Evo-KG. All conclusions should be based on **data available in Evo-KG**, and your responses must be clear, concise, and backed by graph results.

---


**STRICT General Guidelines**:
For `/search_biological_entities` endpoint:
  - Always use this before any other endpoint to fetch general information about the entity.
  - The user asks for a biological entity by its name, id or mentions a term that might match a Gene, Protein, Disease, ChemicalEntity, Phenotype, Tissue, Anatomy, BiologicalProcess, MolecularFunction, CellularComponent, Pathway, Mutation, PMID, Species or PlantExtract by name (e.g., "What diseases are related to 'lung'?" or "Show me tissues containing 'lung'").
  - The user query involves partial or fuzzy matching of names.
  - Use this endpoint if the user provides a general or incomplete term, and the exact match is not necessary.

For '/predict_tail' and '/get_prediction_rank' endpoints:
    -Always use the `/search_biological_entities` endpoint to get the model_id of the head entity before using these endpoints.
    -Always output the scores and briefly tell the user that the scores closer to zero (less negative) indicate a stronger prediction.
    -After getting the predictions, always use the `/search_biological_entities` endpoint to get all the details of the predicted tail entities and display them.
    -Always ensure that the provided head, relation, and tail (if applicable) match the model_id and relationship names as defined in the EvoKG by first using the `/search_biological_entities` endpoint.
    -If the user provides ambiguous or partial input, clarify or guide them to provide exact identifiers before using these endpoints.
    -If the requested entity or relationship is not found in Evo-KG, return an appropriate error message or clarification request rather than invoking the endpoint.

Follow-up: **STRICTLY** FOLLOW THE FOLLOW-UP RESPONSE GUIDELINES.
Large Outputs: For extensive data (e.g., Gene sequence, SMILES), ask users before displaying full details:
"The requested data is large. Display fully or summarize?"
Seek confirmation if the data is large.
Clarity: ALWAYS SPECIFY IF ANSWER IS EvoKG DATA AND GPT GENERATED ("generated by GPT-4o-mini").
Relevance: Limit responses to Evo-KG-related questions or relevant supplementary GPT-4o-mini insights.
Interaction: Keep responses concise and offer summaries or options for large datasets.""".strip()

        super().__init__(*args, **kwargs)

        self.greeting = """
        # Welcome to EvoKG Chatbot

        #### I'm the EvoKG Assistant, and I’m here to help you explore and understand the EvoKG knowledge graph.

        ## Sample Questions You Can Ask
        To get started, try asking questions like:

        * Get details about the disease Stomach Neoplasms.
        * How many nodes are connected to Stomach Neoplasms in EvoKG?
        * Predict new CHEMICALENTITY_DISEASE links for a specific ChemicalEntity.

        These examples highlight how EvoKG can answer specific queries and assist in predictive biological analysis.

        #### Feel free to ask questions, and I’ll do my best to assist you!
        ---
                """

        self.description = "Queries the EvoKG knowledge graph."
        self.avatar = "🧬"
        self.user_avatar = "👤"
        self.name = "EvoKG Assistant"
        
        # Neo4j setup from frontend .env
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")

        self.neo_driver = GraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )


        self.api_base = API_BASE_URL

    # helper function to make API calls
    def api_call(self, endpoint, timeout=60, **kwargs):
        url = f"{self.api_base}/{endpoint}"
        logging.info(f"############ api_call: url={url}, kwargs={kwargs}")
        response = requests.get(url, params=kwargs, timeout=timeout)
        response.raise_for_status()
        return response.json()

    @ai_function
    def hello_world(self) -> dict:
        """
        A simple test endpoint that returns 'Hello, World!'

        Returns:
          dict: A simple greeting message
        """
        try:
            response = self.api_call("hello_world")
            return response
        except Exception as e:
            logging.error(f"Error calling hello_world endpoint: {str(e)}")
            return {"error": f"Failed to get hello world message: {str(e)}"}

    @ai_function
    def get_sample_triples(
        self,
        rel_type: Annotated[
            str,
            AIParam(
                desc="The relationship type to filter triples (e.g. GENE_GENE, GENE_DISEASE, GENE_PHENOTYPE)"
            ),
        ],
    ) -> List[dict]:
        """
        Retrieve sample triples based on the relationship type

        Args:
          rel_type: The relationship type to filter triples. (e.g. GENE_GENE, GENE_DISEASE, GENE_PHENOTYPE)

        Returns:
          List[dict]: A list of triples with head, relation, and tail
        """
        try:
            response = self.api_call("sample_triples", rel_type=rel_type)
            return response
        except Exception as e:
            logging.error(f"Error calling sample_triples endpoint: {str(e)}")
            return {"error": f"Failed to retrieve sample triples: {str(e)}"}

    @ai_function
    def get_nodes_by_label(
        self,
        label: Annotated[
            str,
            AIParam(
                desc="The label of the nodes to retrieve (e.g., Gene, Protein, Disease, ChemicalEntity, Phenotype, Tissue, Anatomy, BiologicalProcess, MolecularFunction, CellularComponent, Pathway, Mutation, PMID, Species, PlantExtract)"
            ),
        ],
    ) -> List[dict]:
        """
        Retrieve 10 nodes of a given type, returning either id or name as available.

        Args:
          label: The label of the nodes to retrieve (e.g., Gene, Protein, Disease, ChemicalEntity, Phenotype, Tissue, Anatomy, BiologicalProcess, MolecularFunction, CellularComponent, Pathway, Mutation, PMID, Species, PlantExtract)

        Returns:
          List[dict]: A list of up to 10 nodes with their primary identifiers
        """
        try:
            response = self.api_call("get_nodes_by_label", label=label)
            return response
        except Exception as e:
            logging.error(f"Error calling get_nodes_by_label endpoint: {str(e)}")
            return {"error": f"Failed to retrieve nodes by label: {str(e)}"}

    @ai_function
    def get_subgraph(
        self,
        property_name: Annotated[
            str,
            AIParam(
                desc="Property name of the start node to search for (e.g., name, id)"
            ),
        ],
        property_value: Annotated[
            str, AIParam(desc="Value of the property to search for")
        ],
        node_label: Annotated[
            str,
            AIParam(desc="Label of the start node to search for (e.g., Gene, Protein)"),
        ],
    ) -> dict:
        """
        Retrieve a subgraph of related nodes by specifying the property and value of the start node

        Args:
            property_name: Property name of the start node to search for (e.g., name, id)
            property_value: Value of the property to search for
            node_label: Label of the start node to search for (e.g., Gene, Protein)

        Returns:
            dict: A subgraph of nodes related to the specified node
        """
        try:
            params = {
                "property_name": property_name,
                "property_value": property_value,
                "node_label": node_label,
            }
            response = self.api_call("subgraph", **params)
            return response
        except Exception as e:
            logging.error(f"Error calling subgraph endpoint: {str(e)}")
            return {"error": f"Failed to retrieve subgraph: {str(e)}"}

    @ai_function
    def search_biological_entities(
        self,
        targetTerm: Annotated[
            str,
            AIParam(
                desc="The name or id or the term to search for in biological entities"
            ),
        ],
    ) -> List[dict]:
        """
        Search biological entities such as  Gene, Protein, Disease, ChemicalEntity, Phenotype, Tissue, Anatomy, BiologicalProcess, MolecularFunction, CellularComponent, Pathway, Mutation, PMID, Species or PlantExtract by name or id

        Args:
          targetTerm: The name or id or the term to search for in biological entities

        Returns:
          List[dict]: A list of entity types with their top 3 matching entities
        """
        try:
            response = self.api_call(
                "search_biological_entities", targetTerm=targetTerm
            )
            return response
        except Exception as e:
            logging.error(
                f"Error calling search_biological_entities endpoint: {str(e)}"
            )
            return {"error": f"Failed to search biological entities: {str(e)}"}

    @ai_function
    def get_entity_relationships(
        self,
        entity_type: Annotated[
            str,
            AIParam(
                desc="The type of entity to search for (e.g., Gene, Protein, Disease)"
            ),
        ],
        property_name: Annotated[
            str,
            AIParam(desc="The property used to identify the entity (e.g., id, name)"),
        ],
        property_value: Annotated[
            str, AIParam(desc="The value of the property for the entity")
        ],
        relationship_type: Annotated[
            str,
            AIParam(
                desc="The type of relationship to filter by (e.g., GENE_DISEASE, PROTEIN_PROTEIN)"
            ),
        ] = None,
    ) -> dict:
        """
        Retrieve the count and list of related entities for a specified entity and optionally by relationship type

        Args:
          entity_type: The type of entity to search for (e.g., Gene, Protein)
          property_name: The property used to identify the entity (e.g., id, name)
          property_value: The value of the property for the entity
          relationship_type: The type of relationship to filter by (optional)

        Returns:
          dict: The count and details of related entities, optionally filtered by relationship type
        """
        try:
            params = {
                "entity_type": entity_type,
                "property_name": property_name,
                "property_value": property_value,
            }
            if relationship_type:
                params["relationship_type"] = relationship_type

            response = self.api_call("entity_relationships", **params)
            return response
        except Exception as e:
            logging.error(f"Error calling entity_relationships endpoint: {str(e)}")
            return {"error": f"Failed to retrieve entity relationships: {str(e)}"}

    @ai_function
    def check_relationship(
        self,
        entity1_type: Annotated[
            str, AIParam(desc="The type of the first entity (e.g., Gene, Protein)")
        ],
        entity1_property_name: Annotated[
            str,
            AIParam(
                desc="The property name to identify the first entity (e.g., id, name)"
            ),
        ],
        entity1_property_value: Annotated[
            str, AIParam(desc="The property value to identify the first entity")
        ],
        entity2_type: Annotated[
            str, AIParam(desc="The type of the second entity (e.g., Disease, Protein)")
        ],
        entity2_property_name: Annotated[
            str,
            AIParam(
                desc="The property name to identify the second entity (e.g., id, name)"
            ),
        ],
        entity2_property_value: Annotated[
            str, AIParam(desc="The property value to identify the second entity")
        ],
    ) -> dict:
        """
        Check if a relationship exists between two entities and return the type of relationship

        Args:
          entity1_type: The type of the first entity (e.g., Gene, Protein)
          entity1_property_name: The property name to identify the first entity (e.g., id, name)
          entity1_property_value: The property value to identify the first entity
          entity2_type: The type of the second entity (e.g., Disease, Protein)
          entity2_property_name: The property name to identify the second entity (e.g., id, name)
          entity2_property_value: The property value to identify the second entity

        Returns:
          dict: Information whether a relationship exists and its type
        """
        try:
            params = {
                "entity1_type": entity1_type,
                "entity1_property_name": entity1_property_name,
                "entity1_property_value": entity1_property_value,
                "entity2_type": entity2_type,
                "entity2_property_name": entity2_property_name,
                "entity2_property_value": entity2_property_value,
            }
            response = self.api_call("check_relationship", **params)
            return response
        except Exception as e:
            logging.error(f"Error calling check_relationship endpoint: {str(e)}")
            return {"error": f"Failed to check relationship: {str(e)}"}

    @ai_function
    def predict_tail(
        self,
        head: Annotated[
            str, AIParam(desc="model_id for the head entity for the prediction")
        ],
        relation: Annotated[
            str,
            AIParam(desc="Relation for the prediction"),
        ],
        top_k_predictions: Annotated[
            int, AIParam(desc="Number of top predictions to return (default is 10)")
        ] = 10,
    ) -> dict:
        """
        Predict the top K tail entities given 'model_id' of entities and relation using a PyKEEN KGE model

        Args:
          head: model_id for the head entity for the prediction
          relation: Relation for the prediction
          top_k_predictions: Number of top predictions to return (default is 10)

        Returns:
          dict: Head entity, relation, and a list of predicted tail entities with scores
        """
        try:
            params = {
                "head": head,
                "relation": relation,
                "top_k_predictions": top_k_predictions,
            }
            response = self.api_call("predict_tail", **params)
            return response
        except Exception as e:
            logging.error(f"Error calling predict_tail endpoint: {str(e)}")
            return {"error": f"Failed to predict tail entities: {str(e)}"}

    @ai_function
    def get_prediction_rank(
        self,
        head: Annotated[
            str, AIParam(desc="model_id for head entity for the prediction")
        ],
        relation: Annotated[
            str,
            AIParam(desc="Relation for the prediction"),
        ],
        tail: Annotated[
            str, AIParam(desc="model_id for tail entity to check for its rank")
        ],
    ) -> dict:
        """
        Get the rank and score of a specific tail entity for a given head and relation, along with the maximum score.

        Args:
          head: model_id for head entity for the prediction
          relation: Relation for the prediction
          tail: model_id for tail entity to check for its rank

        Returns:
          dict: The rank, score, and maximum score of the prediction
        """
        try:
            params = {"head": head, "relation": relation, "tail": tail}
            response = self.api_call("get_prediction_rank", **params)
            return response
        except Exception as e:
            logging.error(f"Error calling get_prediction_rank endpoint: {str(e)}")
            return {"error": f"Failed to get prediction rank: {str(e)}"}


    @ai_function
    def get_nodes(
        self,
        term1: Annotated[str, AIParam(desc="The first biological term (e.g., metformin)")],
        term2: Annotated[str, AIParam(desc="The second biological term (e.g., Parkinson's disease)")],
    ) -> dict:
        """
        Search for all possible nodes present in the EvoKg database for the extracted entities.
        """
        try:
            
            # Step 1: Search for biological entities for both terms
            term1_groups = self.search_biological_entities(term1)
            term2_groups = self.search_biological_entities(term2)

            def extract_entities(groups):
                entities = []
                for group in groups:
                    entity_type = group.get("entityType")
                    for ent in group.get("topEntities", []):
                        props = ent.get("properties", {})
                        entities.append({
                            "model_id": props.get("model_id"),
                            "type": entity_type,
                            "name": props.get("name"),
                            "id": props.get("id")
                        })
                return entities

            term1_entities = extract_entities(term1_groups)
            term2_entities = extract_entities(term2_groups)
            
            
            
            return {
                "term1": term1,
                "term2": term2,
                "entities_found": {
                    "term1_entities": term1_entities,
                    "term2_entities": term2_entities,
                },
            }
            

        except Exception as e:
            logging.error(f"Error in hypothesis evaluation: {e}")
            return {"error": f"Failed to evaluate hypothesis: {e}"}
        
    @ai_function
    def check_direct_relations(
        self,
        term1_entities: Annotated[
            list[dict],
            AIParam(desc="List of entities from term1, each a dict with keys: model_id, id, name, type")
        ],
        term2_entities: Annotated[
            list[dict],
            AIParam(desc="List of entities from term2, each a dict with keys: model_id, id, name, type")
        ]
    ) -> dict:
        """
        Checks for direct relationships between all entity pairs from term1 and term2, and returns a ranked summary.
        """
        try:
            results = []
            term1_rel_count = {e["model_id"]: 0 for e in term1_entities}
            term2_rel_count = {e["model_id"]: 0 for e in term2_entities}

            for ent1 in term1_entities:
                for ent2 in term2_entities:
                    try:
                        relationship_result = self.check_relationship(
                            entity1_type=ent1["type"],
                            entity1_property_name="model_id",
                            entity1_property_value=ent1["model_id"],
                            entity2_type=ent2["type"],
                            entity2_property_name="model_id",
                            entity2_property_value=ent2["model_id"],
                        )
                        if isinstance(relationship_result, str):
                            relationship_result = json.loads(relationship_result)

                        if relationship_result.get("exists"):
                            try:
                                score_result = self.get_prediction_rank(
                                    head=ent1["model_id"],
                                    relation=relationship_result.get("relationship_type"),
                                    tail=ent2["model_id"]
                                )
                                if isinstance(score_result, str):
                                    score_result = json.loads(score_result)
                            except Exception as e:
                                logging.warning(f"Prediction score failed for {ent1['name']} → {ent2['name']}: {e}")
                                score_result = {"score": None, "rank": None, "max_score": None}

                            results.append({
                                "term1_entity": ent1,
                                "term2_entity": ent2,
                                "relationship_type": relationship_result.get("relationship_type"),
                                "score": score_result.get("score"),
                                "rank": score_result.get("rank"),
                                "max_score": score_result.get("max_score")
                            })
                            term1_rel_count[ent1["model_id"]] += 1
                            term2_rel_count[ent2["model_id"]] += 1

                    except Exception as e:
                        logging.warning(f"Failed relationship check for {ent1['name']} → {ent2['name']}: {e}")
                        continue

            # Step 5: Identify Top Connected Nodes
            max_term1 = max(term1_rel_count.values(), default=0)
            max_term2 = max(term2_rel_count.values(), default=0)

            top_term1_nodes = [
                {
                    "model_id": e.get("model_id"),
                    "id": e.get("id"),
                    "name": e.get("name"),
                    "type": e.get("type"),
                    "relation_count": term1_rel_count.get(e.get("model_id"), 0)
                }
                for e in term1_entities if term1_rel_count[e["model_id"]] == max_term1 and max_term1 > 0
            ]
            top_term2_nodes = [
                {
                    "model_id": e.get("model_id"),
                    "id": e.get("id"),
                    "name": e.get("name"),
                    "type": e.get("type"),
                    "relation_count": term2_rel_count.get(e.get("model_id"), 0)
                }
                for e in term2_entities if term2_rel_count[e["model_id"]] == max_term2 and max_term2 > 0
            ]

            return {
                "evaluated_pairs": len(results),
                "results": results,
                "relationship_summary": {
                    "total_relationships_found": len(results),
                    "top_term1_nodes": top_term1_nodes,
                    "top_term2_nodes": top_term2_nodes
                }
            }

        except Exception as e:
            logging.error(f"Error in evaluating relationships: {e}")
            return {"error": f"Failed to evaluate relationships: {e}"}
     
    @ai_function
    def get_k_shortest_paths(
        self,
        head_model_id: Annotated[str, AIParam(desc="model_id of head entity")],
        tail_model_id: Annotated[str, AIParam(desc="model_id of tail entity")],
        k: Annotated[int, AIParam(desc="number of valid paths to return")] = 3,
        must_include_label: Annotated[str, AIParam(desc="Node label that must appear in path (e.g., 'Gene'). Leave blank to disable")] = ""
    ) -> dict:
        """
        Finds up to k valid paths between two nodes:
        - Filters out unwanted relationships (like PMID_*, etc.)
        - Can enforce inclusion of a specific node label (e.g., Gene) if specified
        - Returns readable paths (node names or model_ids + relationship type)
        """
        try:
            query = """
            MATCH (start {model_id: $head_model_id}), (end {model_id: $tail_model_id})
            CALL {
            CALL db.relationshipTypes() YIELD relationshipType
            WITH collect(relationshipType) AS allRels
            WITH [r IN allRels WHERE NOT r IN [
                'Species_AssociatedWith',
                'PMID_CellularComponent',
                'PMID_ChemicalEntity',
                'PMID_Disease',
                'PMID_Protein',
                'PMID_Tissue'
            ]] AS whiteList
            RETURN apoc.text.join(whiteList, '|') AS relFilter
            }
            CALL apoc.path.expandConfig(start, {
            endNodes: [end],
            relationshipFilter: relFilter,
            maxLevel: 7,
            bfs: true,
            uniqueness: "NODE_PATH",
            limit: 100
            })
            YIELD path
            WITH path
            """ + (
                "WHERE any(n IN nodes(path) WHERE $must_include_label IN labels(n))\n" if must_include_label else ""
            ) + """
            RETURN path
            LIMIT $k
            """

            with self.neo_driver.session() as session:
                result = session.run(query, {
                    "head_model_id": head_model_id,
                    "tail_model_id": tail_model_id,
                    "k": k,
                    "must_include_label": must_include_label
                })

                paths = []
                for record in result:
                    rel_path = record["path"].relationships
                    node_path = record["path"].nodes
                    steps = []
                    for i in range(len(rel_path)):
                        from_node = node_path[i]
                        to_node = node_path[i + 1]
                        steps.append({
                            "from": from_node.get("name", from_node.get("model_id", "")),
                            "to": to_node.get("name", to_node.get("model_id", "")),
                            "relation": rel_path[i].type
                        })
                    paths.append(steps)

                return {"paths": paths}

        except Exception as e:
            return {"error": f"Neo4j path query failed: {str(e)}"}

    @ai_function
    def batch_check_relationships(
        self,
        term1_entities: Annotated[list[dict], AIParam(desc="List of Term1 entities with model_id")],
        term2_entities: Annotated[list[dict], AIParam(desc="List of Term2 entities with model_id")]
    ) -> list[dict]:
        """
        Efficiently check all relationships between term1 and term2 entities using a single Neo4j Cypher query.

        Returns:
            List of dicts with term1_id, term2_id, and relationship type
        """
        try:
            term1_ids = [e["model_id"] for e in term1_entities]
            term2_ids = [e["model_id"] for e in term2_entities]

            cypher = """
            UNWIND $term1_ids AS id1
            UNWIND $term2_ids AS id2
            MATCH (a {model_id: id1})-[r]-(b {model_id: id2})
            RETURN a.model_id AS term1_id, type(r) AS relationship_type, b.model_id AS term2_id
            """

            with self.neo_driver.session() as session:
                result = session.run(cypher, {
                    "term1_ids": term1_ids,
                    "term2_ids": term2_ids
                })
                
                records = result.data()

                
            # Build fast lookup maps
            term1_dict = {e["model_id"]: e for e in term1_entities}
            term2_dict = {e["model_id"]: e for e in term2_entities}
            term1_rel_count = {e["model_id"]: 0 for e in term1_entities}
            term2_rel_count = {e["model_id"]: 0 for e in term2_entities}

            results = []

            for row in records:
                t1_id = row["term1_id"]
                t2_id = row["term2_id"]
                rel_type = row["relationship_type"]

                ent1 = term1_dict.get(t1_id)
                ent2 = term2_dict.get(t2_id)

                if not ent1 or not ent2:
                    continue

                results.append({
                    "term1_entity": ent1,
                    "term2_entity": ent2,
                    "relationship_type": rel_type,
                    "score": None,
                    "rank": None,
                    "max_score": None
                })

                term1_rel_count[t1_id] += 1
                term2_rel_count[t2_id] += 1

            # Compute top connected nodes
            max_term1 = max(term1_rel_count.values(), default=0)
            max_term2 = max(term2_rel_count.values(), default=0)

            top_term1_nodes = [
                {
                    **term1_dict[mid],
                    "relation_count": count
                }
                for mid, count in term1_rel_count.items()
                if count == max_term1 and count > 0
            ]

            top_term2_nodes = [
                {
                    **term2_dict[mid],
                    "relation_count": count
                }
                for mid, count in term2_rel_count.items()
                if count == max_term2 and count > 0
            ]

            return {
                "evaluated_pairs": len(results),
                "results": results,
                "relationship_summary": {
                    "total_relationships_found": len(results),
                    "top_term1_nodes": top_term1_nodes,
                    "top_term2_nodes": top_term2_nodes
                }
            }

        except Exception as e:
            logging.error(f"Error in batch_check_relationships: {e}")
            return {"error": f"Failed to run batch relationship check: {e}"}
        

    @ai_function
    def evaluate_hypothesis(
        self,
        term1: Annotated[str, AIParam(desc="First biological term (head)")],
        term2: Annotated[str, AIParam(desc="Second biological term (tail)")],
        other_terms: Annotated[list[str], AIParam(desc="Other relevant biological concepts mentioned in hypothesis")]
    ) -> dict:
        """
        Run hypothesis testing between two terms using entity search, direct relation check, prediction, and explanation.
        """
        try:
            logging.info(f"--- Starting hypothesis evaluation: {term1} ↔ {term2} ---")

            output = {
                "term1": term1,
                "term2": term2,
                "entities_found": {},
                "direct_relation_found": False,
                "direct_relation_details": [],
                "prediction_results": [],
                "shortest_paths": []
            }

            # Step 1: Entity search
            t0 = time.time()
            logging.info("🔍 Fetching nodes for term1 and term2...")
            nodes_result = self.get_nodes(term1, term2)
            logging.info(f"🔍 get_nodes() took {time.time() - t0:.2f}s")
            
            if "error" in nodes_result:
                logging.error("❌ Error in get_nodes.")
                return nodes_result

            term1_entities = nodes_result["entities_found"].get("term1_entities", [])
            term2_entities = nodes_result["entities_found"].get("term2_entities", [])

            logging.info(f"✅ Found {len(term1_entities)} nodes for term1: '{term1}'")
            logging.info(f"✅ Found {len(term2_entities)} nodes for term2: '{term2}'")

            output["entities_found"] = {
                "term1_entities": term1_entities,
                "term2_entities": term2_entities
            }

            # Step 2.1: No nodes at all
            if not term1_entities and not term2_entities:
                logging.warning("🚫 No nodes found for either term.")
                return {
                    "status": "no_nodes",
                    "message": f" no nodes for either '{term1}' or '{term2}' exist in our database."
                }

            # Step 2.2: Partial match
            if not term1_entities or not term2_entities:
                missing_term = term1 if not term1_entities else term2
                found_term = term2 if not term1_entities else term1
                found_nodes = term2_entities if not term1_entities else term1_entities
                t1 = time.time()

                logging.warning(f"⚠️ Partial match: '{missing_term}' not found.{time.time() - t1:.2f}s")
                similar_terms = self.search_biological_entities(missing_term)
                return {
                    "status": "partial_nodes",
                    "message": f"We found nodes for '{found_term}' but none for '{missing_term}'.",
                    "similar_node_suggestions": similar_terms,
                    "found_nodes": found_nodes,
                    "missing_term": missing_term,
                }

            # Step 3: Direct relationship check
            logging.info("🔗 Checking direct relationships between node pairs...")
            t2 = time.time()
            direct_rel_result = self.check_direct_relations(term1_entities, term2_entities)
            # direct_rel_result = self.batch_check_relationships(term1_entities, term2_entities)

            logging.info(f"🔗 batch_check_relationships() took {time.time() - t2:.2f}s")

            if "error" in direct_rel_result:
                logging.error("❌ Error during check_direct_relations.")
                return direct_rel_result

            output["direct_relation_summary"] = direct_rel_result.get("relationship_summary", {})
            output["direct_relation_details"] = direct_rel_result.get("results", [])
            logging.info(f"✅ Direct relationship pairs found: {len(direct_rel_result.get('results', []))}")

            if direct_rel_result["evaluated_pairs"] > 0:
                output["direct_relation_found"] = True
                output["source_confidence"] = "100% fact from database"

                top_term1_nodes = direct_rel_result["relationship_summary"].get("top_term1_nodes", [])
                top_term2_nodes = direct_rel_result["relationship_summary"].get("top_term2_nodes", [])

                if not top_term1_nodes or not top_term2_nodes:
                    logging.warning("⚠️ No top term1 or term2 nodes found despite relationships.")
                    return output

                # Use the first top node from each term
                top_term1 = top_term1_nodes[0]
                top_term2 = top_term2_nodes[0]

                head = top_term1["model_id"]
                tail = top_term2["model_id"]
                relation = f"{top_term1['type']}_{top_term2['type']}"
                relation_rev = f"{top_term2['type']}_{top_term1['type']}"

                logging.info(f"🎯 Top direct relation nodes for prediction: {head} --[{relation}]--> {tail}")

                # Tail prediction: head → relation → ?
                t3 = time.time()
                tail_predictions = []
               
                try:
                    raw_preds = self.predict_tail(head=head, relation=relation)
                    preds = raw_preds.get("predictions", [])[:3]
                    tail_predictions.extend(preds)
                    logging.info(f"🔮 predict_tail() took {time.time() - t3:.2f}s")
                except Exception as e:
                    logging.error(f"❌ Failed to fetch tail predictions from {head} via {relation}: {e}")
                # Reverse prediction: tail → relation_rev → ?
                reverse_predictions = []
                t4 = time.time()
                try:
                    raw_rev_preds = self.predict_tail(head=tail, relation=relation_rev)
                    rev_preds = raw_rev_preds.get("predictions", [])[:3]
                    reverse_predictions.extend(rev_preds)
                    logging.info(f"🔁 reverse predict_tail() took {time.time() - t4:.2f}s")
                except Exception as e:
                    logging.error(f"❌ Failed to fetch reverse predictions from {tail} via {relation_rev}: {e}")

                output["prediction_results"] = {
                    "tail_predictions": tail_predictions,
                    "reverse_tail_predictions": reverse_predictions,
                    "head": top_term1,
                    "tail": top_term2
                }

                return output
            


            else:
                logging.info("❌ No direct relationship found. ")
                output["direct_relation_found"] = False

                # Step 4: No direct relation — get shortest paths
                t5 = time.time()
                best_head = term1_entities[0]["model_id"]
                best_tail = term2_entities[0]["model_id"]

                try:
                    logging.info(f"🧬 Attempting shortest path search between {best_head} and {best_tail}")
                    k_paths = self.get_k_shortest_paths(
                        head_model_id=best_head,
                        tail_model_id=best_tail,
                        k=3
                    )
                    output["shortest_paths"] = k_paths.get("paths", [])
                    logging.info(f"✅ Shortest paths found: {len(output['shortest_paths'])}")
                    logging.info(f"🧬 get_k_shortest_paths() took {time.time() - t5:.2f}s")

                except Exception as e:
                    logging.warning(f"⚠️ Neo4j shortest path failed: {e}")
                    output["shortest_paths"] = []

                # Initialize prediction results structure
                t6 = time.time()
                output["prediction_results"] = {
                    "head": term1_entities[0],
                    "tail": term2_entities[0],
                    "fallback_predictions": {
                        "term1_top_predictions": [],  # Head → Tail predictions
                        "term2_top_predictions": []   # Tail → Head predictions
                    }
                }

                 # Step 5: Predict tails from head → ?
                best_head_entity = term1_entities[0]
                best_tail_entity = term2_entities[0]
                best_head = best_head_entity["model_id"]
                best_tail = best_tail_entity["model_id"]
                relation = f"{best_head_entity['type']}_{best_tail_entity['type']}"
                relation_rev = f"{best_tail_entity['type']}_{best_head_entity['type']}"

                try:
                    logging.info(f"🔮 Predicting tails from {best_head} via {relation}")
                    pred = self.predict_tail(head=best_head, relation=relation, top_k_predictions=10)
                    preds = pred.get("predictions", [])

                    matched_pred = next((p for p in preds if p.get("model_id") == best_tail), None)

                    if matched_pred:
                        logging.info(f"🎯 Matched tail found in predictions. Getting rank info...")
                        try:
                            rank_info = self.get_prediction_rank(head=best_head, relation=relation, tail=best_tail)
                            matched_pred["rank"] = rank_info.get("rank")
                            matched_pred["score"] = rank_info.get("score")
                            matched_pred["max_score"] = rank_info.get("max_score")
                            logging.info(f"🎯 Tail prediction match found in {time.time() - t6:.2f}s")
                        except Exception as e:
                            logging.warning(f"⚠️ Failed to fetch rank for matched prediction: {e}")

                        output["prediction_results"] = {
                            "tail_prediction_match": matched_pred,
                            "head": term1_entities[0],
                            "tail": term2_entities[0]
                        }
                    else:
                        logging.info("🔁 No match found in predicted tails. Showing top 3 fallback predictions.")
                        logging.info(f"🔮 No tail match — fallback top predictions computed in {time.time() - t6:.2f}s")
                        output["prediction_results"] = {
                            "fallback_predictions": {
                                "term1_top_predictions": preds[:3]
                            },
                            "head": term1_entities[0],
                            "tail": term2_entities[0]
                        }

                except Exception as e:
                    logging.warning(f"⚠️ Failed to predict tails from head: {e}")
                    output["prediction_results"] = {
                        "fallback_predictions": {
                            "term1_top_predictions": []
                        },
                        "head": term1_entities[0],
                        "tail": term2_entities[0]
                    }

                # Step 6: Reverse tail predictions (tail → ?)
                t7 = time.time()
                try:
                    logging.info(f"🔮 Predicting reverse tails from {best_tail} via {relation_rev}")
                    rev_pred = self.predict_tail(head=best_tail, relation=relation_rev)
                    output["prediction_results"].setdefault("fallback_predictions", {})
                    output["prediction_results"]["fallback_predictions"]["term2_top_predictions"] = rev_pred.get("predictions", [])[:3]
                    logging.info(f"🔁 fallback reverse predict_tail() took {time.time() - t7:.2f}s")
                except Exception as e:
                    logging.warning(f"⚠️ Reverse fallback prediction failed: {e}")
                    output["prediction_results"].setdefault("fallback_predictions", {})
                    output["prediction_results"]["fallback_predictions"]["term2_top_predictions"] = []

                return output
        
        except Exception as e:
            logging.error(f"❌ Critical failure in evaluate_hypothesis: {e}")
            return {"error": f"Failed to evaluate hypothesis: {e}"}
    
    logging.info("--- Hypothesis evaluation complete ---")
        
            
    @ai_function
    def render_response(output: dict) -> str:
        """Render a human-readable explanation of the hypothesis test outcome."""

        if "error" in output:
            return f"❌ Error: {output['error']}"

        lines = []

        # Case 1: No nodes for both terms
        if output.get("status") == "no_nodes":
            return f" Unfortunately, No nodes found for either '{output['term1']}' or '{output['term2']}'. Try different terms or rephrase your hypothesis. Would you like to try a different term or explore related entities? Feel free to provide more context, and I'll assist you further."

        # Case 2: One term missing
        if output.get("status") == "partial_nodes":
            missing = output["missing_term"]
            found = output.get("found_nodes", [])
            suggestions = output.get("similar_node_suggestions", [])
            lines.append(f"🟨 Partial match: Found nodes for one term, but **'{missing}'** was not found.\n")

            lines.append("### ✅ Found Nodes:")
            for node in found:
                lines.append(f"- {node['name']} ({node['type']})")

            if suggestions:
                lines.append("\n### 💡 Suggestions for the missing term:")
                for s in suggestions:
                    lines.append(f"- {s}")
            else:
                lines.append("\n⚠️ No similar suggestions found.")

            return "\n".join(lines)

        # Case 3: Both terms have nodes — Show Entity Summary
        lines.append("### ✅ Entities Found\n")
        key_map = {"term1_entities": "Term 1", "term2_entities": "Term 2"}
        for key, ents in output.get("entities_found", {}).items():
            lines.append(f"**{key_map.get(key, key)} Entities:**")
            if not ents:
                lines.append("- No entities found.")
            else:
                sorted_ents = sorted(ents, key=lambda x: (x['type'], x['name']))
                for ent in sorted_ents[:10]:
                    lines.append(f"- {ent['name']} ({ent['type']})")
                if len(ents) > 10:
                    lines.append(f"... and {len(ents) - 10} more entities found.")
            lines.append("")

        # === Direct Relationship Case ===
        if output.get("direct_relation_found"):
            lines.append("---\n### 🔗 100% Fact from Database: Direct Relationship Found\n")

            # ➤ Detailed Relationship Info
            details = output.get("direct_relation_details", [])
            if details:
                grouped = {}
                for rel in details:
                    rel_type = rel.get("relationship_type", "Unknown")
                    grouped.setdefault(rel_type, []).append(rel)

                for rel_type, group in grouped.items():
                    lines.append(f"#### 🧬 Relationship Type: `{rel_type}`")
                    for i, rel in enumerate(group[:10], 1):
                        ent1 = rel['term1_entity']
                        ent2 = rel['term2_entity']
                        lines.append(f"{i}. **{ent1['name']}** ({ent1['type']}) → **{ent2['name']}** ({ent2['type']})")
                        if rel.get("score") is not None:
                            lines.append(f"   └─ Score: {rel['score']} (Rank {rel['rank']}/{rel['max_score']})")
                    lines.append("")

            # ➤ Summary
            summary = output.get("direct_relation_summary", {})
            if summary:
                lines.append("### 📊 Relationship Summary\n")
                for side in ["top_term1_nodes", "top_term2_nodes"]:
                    label = "Term 1" if "term1" in side else "Term 2"
                    nodes = summary.get(side, [])
                    if nodes:
                        lines.append(f"**Top Connected {label} Entities:**")
                        for ent in nodes[:5]:
                            lines.append(f"- {ent['name']} ({ent['type']}) — Relations: {ent['relation_count']}")
                lines.append("")

            # ➤ Tail predictions to support direct link
            preds = output.get("prediction_results", {})
            if preds:
                # Tail predictions
                tail_preds = preds.get("tail_predictions", [])
                if tail_preds:
                    lines.append("### 🤖 Top Predicted Tails from Head\n")
                    for i, p in enumerate(tail_preds[:3], 1):
                        lines.append(f"{i}. {p.get('name')} — Score: {p.get('score')}")
                else:
                    lines.append("No predictions found")

                # ➤ Reverse predictions
                rev_preds = preds.get("reverse_tail_predictions", [])
                if rev_preds:
                    lines.append("\n### 🔁 Top Predicted Tails from Tail\n")
                    for i, p in enumerate(rev_preds[:3], 1):
                        lines.append(f"{i}. {p.get('name')} — Score: {p.get('score')}")
                else:
                    lines.append(" No reverse predictions found.")


        # === No Direct Relationship Case ===
        else:
            lines.append("---\n### ❌ No Direct Relationship Found\n")

            # ➤ Show K-shortest paths
            all_paths = output.get("shortest_paths", [])
            if all_paths:
                lines.append("\n### 🧬 K-Shortest Biological Paths\n")
                for i, path in enumerate(all_paths, 1):
                    path_str = " → ".join(f"{step['from']} --[{step['relation']}]→ {step['to']}" for step in path)
                    lines.append(f"{i}. {path_str}")
            else:
                lines.append("🔍 No paths found.")

            # ➤ Indirect Predictions
            preds = output.get("prediction_results", {})
            if preds:
                # Check for matched prediction
                matched_pred = preds.get("tail_prediction_match")
                if matched_pred:
                    lines.append("\n### 🎯 Prediction Match Found\n")
                    lines.append(f"**{preds['head']['name']}** → **{preds['tail']['name']}**")
                    lines.append(f"- Score: {matched_pred.get('score')}")
                    lines.append(f"- Rank: {matched_pred.get('rank')}/{matched_pred.get('max_score')}")
                
                # Fallback predictions
                fallback = preds.get("fallback_predictions", {})
                if fallback:
                    lines.append("\n### 🔮 Top Predictions from Each Term\n")
                    
                    # Term 1 predictions
                    term1_preds = fallback.get("term1_top_predictions", [])
                    if term1_preds:
                        lines.append("**From Term 1:**")
                        for i, p in enumerate(term1_preds[:3], 1):
                            lines.append(f"{i}. {p.get('tail_entity')} — Score: {p.get('score')}")
                    else:
                        lines.append("**From Term 1:** No predictions found.")
                    
                    # Term 2 predictions
                    term2_preds = fallback.get("term2_top_predictions", [])
                    if term2_preds:
                        lines.append("\n**From Term 2:**")
                        for i, p in enumerate(term2_preds[:3], 1):
                            lines.append(f"{i}. {p.get('tail_entity')} — Score: {p.get('score')}")
                    else:
                        lines.append("\n**From Term 2:** No predictions found.")

        # === Final Summary ===
        lines.append("\n---\n### 🧠 Final Summary\n")
        if output.get("direct_relation_found"):
            lines.append("✅ A direct relationship was found in the database and is supported by prediction evidence.")
        else:
            lines.append("🔄 While no direct link was found, shortest paths and prediction-based evidence may suggest a latent biological relationship.")
            if all_paths:
                lines.append("- Paths exist between the terms")
            if preds and (preds.get("tail_prediction_match") or preds.get("fallback_predictions")):
                lines.append("- Prediction models show potential connections")
            if not all_paths and not preds:
                lines.append("- No supporting evidence found in our knowledge base")
        return "\n".join(lines)
