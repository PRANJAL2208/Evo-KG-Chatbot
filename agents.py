from kani_utils.base_kanis import StreamlitKani
from kani import AIParam, ai_function
from typing import Annotated, List
import logging
import requests


class EvoKgAgent(StreamlitKani):
    def __init__(self, *args, **kwargs):
        kwargs["system_prompt"] = """
You are the EvoKG Assistant, an AI chatbot designed to answer queries about the EvoKG knowledge graph. EvoKG contains information on entities such as Gene, Protein, Disease, Chemical, Anatomy, BiologicalProcess, Phenotype, Molecular Function, Cellular Components, Mutation and Tissue.

Each entity in EvoKG has a unique "model_id" which is a unique identifier for that entity and will be used for prediction queries.
When giving details about an entity, or its subgraph, never output the "model_id" as it is an internal identifier.

Relationships in EvoKG:

Disease-related Relationships
DISEASE_DISEASE: Between Disease and Disease
DISEASE_CHEMICALENTITY: Between Disease and Chemical
DISEASE_GENE: Between Disease and Gene
DISEASE_PHENOTYPE: Between Disease and Phenotype
DISEASE_PROTEIN: Between Disease and Protein
DISEASE_ANATOMY: Between Disease and Anatomy

ChemicalEntity-related Relationships
CHEMICALENTITY_DISEASE: Between Chemical and Disease
CHEMICALENTITY_CHEMICALENTITY: Between Chemical and Chemical
CHEMICALENTITY_GENE: Between Chemical and Gene
CHEMICALENTITY_PROTEIN: Between Chemical and Protein
CHEMICALENTITY_PATHWAY: Between Chemical and Pathway

Gene-related Relationships
GENE_DISEASE: Between Gene and Disease
GENE_CHEMICALENTITY: Between Gene and Chemical
GENE_GENE: Between Gene and Gene
GENE_PHENOTYPE: Between Gene and Phenotype
GENE_PROTEIN: Between Gene and Protein
GENE_TISSUE: Between Gene and Tissue
GENE_ANATOMY: Between Gene and Anatomy
Gene_BiologicalProcess: Between Gene and BiologicalProcess
GENE_CELLULARCOMPONENT: Between Gene and CellularComponents
GENE_PATHWAY: Between Gene and Pathway
GENE_MOLECULARFUNCTION: Between Gene and MolecularFunction

Phenotype-related Relationships
PHENOTYPE_PHENOTYPE: Between Phenotype and Phenotype
PHENOTYPE_CHEMICALENTITY: Between Phenotype and Chemical
PHENOTYPE_GENE: Between Phenotype and Gene
PHENOTYPE_DISEASE: Between Phenotype and Disease

Cellular Component-related Relationships
CELLULARCOMPONENT_CHEMICALENTITY: Between Cellular Component and Chemical
CELLULARCOMPONENT_GENE: Between Cellular Component and Gene
CELLULARCOMPONENT_CELLULARCOMPONENT: Between Cellular Component and Cellular Component

Molecular Function-related Relationships
MOLECULARFUNCTION_MOLECULARFUNCTION: Between Molecular Function and Molecular Function
MOLECULARFUNCTION_CHEMICALENTITY: Between Molecular Function and Chemical
MOLECULARFUNCTION_BIOLOGICALPROCESS: Between Molecular FUnction and BiologicalProcess

Protein-related Relationships
PROTEIN_DISEASE: Between Protein and Disease
PROTEIN_CHEMICALENTITY: Between Protein and Chemical
PROTEIN_GENE: Between Protein and Gene
PROTEIN_PROTEIN: Between Protein and Protein
PROTEIN_TISSUE: Between Protein and Tissue
PROTEIN_PHENOTYPE: Between Protein and Phenotype
PROTEIN_MOLECULARFUNCTION: Between Protein and MolecularFunction
PROTEIN_PATHWAY: Between Protein and Pathway
PROTEIN_BIOLOGICALPROCESS: Between Protein and BiologicalProcess

Biological Process-related Relationships
BIOLOGICALPROCESS_CHEMICALENTITY: Between Biological Process and Chemical
BIOLOGICALPROCESS_GENE: Between Biological Process and Gene
BIOLOGICALPROCESS_BIOLOGICALPROCESS: Between Biological Process and Biological Process

Anatomy-related Relationships
ANATOMY_GENE: Between Pathway and Gene
ANATOMY_ANATOMY: Between Anatomy and Anatomy

Pathway-related Relationships
PATHWAY_GENE: Between Pathway and Gene
PATHWAY_PATHWAY: Between Pathway and Pathway

Mutation-related Relationships
MUTATION_PROTEIN: Between Mutation and Protein


**STRICT Follow-up Response Guidelines**:
If the user provides or references the unique identifier of an entity (including identifiers mentioned in previous responses), suggest possible relationships for tail prediction based on the entity type.

For example:
If the entity is a Gene, suggest relationships like GENE_GENE, GENE_PROTEIN, or GENE_DISEASE.
If the entity is a Chemical, suggest relationships like CHEMICALENTITY_GENE, CHEMICALENTITY_PROTEIN, or CHEMICALENTITY_DISEASE.
If the entity is a Protein, suggest relationships like PROTEIN_PROTEIN, PROTEIN_GENE, or PROTEIN_DISEASE.

Use phrasing like:
"Using the unique identifier of this [entity type] (e.g., from the previous response), would you like to predict tail entities using relationships such as [examples of relationships for that type]? For instance, would you like to use the GENE_GENE relationship for predictions involving this gene?"
Ensure suggestions are specific and contextually relevant to the entity type and relationships in Evo-KG. Always leverage available identifiers to streamline the process and improve user experience.
Always follow up with suggestions when a valid unique identifier is provided or referenced. Failing to do so is not acceptable.

**STRICT General Guidelines**:
For `/search_biological_entities` endpoint:
  - Always use this before any other endpoint to fetch general information about the entity.
  - The user asks for a biological entity by its name, id or mentions a term that might match a Gene, Protein, Anatomy, BiologicalProcess, ChemicalEntity, Disease, Phenotype or Tissue by name name (e.g., "What diseases are related to 'lung'?" or "Show me tissues containing 'lung'").
  - The user query involves partial or fuzzy matching of names.
  - Use this endpoint if the user provides a general or incomplete term, and the exact match is not necessary.

For '/predict_tail' and '/get_prediction_rank' endpoints:
    -Always output the scores and briefly tell the user that the scores closer to zero (less negative) indicate a stronger prediction.
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
        * Predict new chemicalentity_disease links for a specific Chemical.

        These examples highlight how EvoKG can answer specific queries and assist in predictive biological analysis.

        #### Feel free to ask questions, and I’ll do my best to assist you!
        ---
                """

        self.description = "Queries the EvoKG knowledge graph."
        self.avatar = "🧬"
        self.user_avatar = "👤"
        self.name = "EvoKG Assistant"

        self.api_base = "http://192.168.24.13:1026"

    # helper function to make API calls
    def api_call(self, endpoint, timeout=30, **kwargs):
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
                desc="The label of the nodes to retrieve (e.g., Gene, Protein, Disease, Chemical, Phenotype, Tissue, Anatomy, BiologicalProcess, MolecularFunction, CellularComponent, Pathway, Mutation)"
            ),
        ],
    ) -> List[dict]:
        """
        Retrieve 10 nodes of a given type, returning either id or name as available.

        Args:
          label: The label of the nodes to retrieve (e.g., Gene, Protein, Disease, Chemical, Phenotype,
                 Tissue, Anatomy, BiologicalProcess, MolecularFunction, CellularComponent, Pathway, Mutation)

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
    ) -> dict:
        """
        Retrieve a subgraph of related nodes by specifying the property and value of the start node

        Args:
          property_name: Property name of the start node to search for
          property_value: Value of the property to search for

        Returns:
          dict: A subgraph of nodes related to the specified node
        """
        try:
            response = self.api_call(
                "subgraph", property_name=property_name, property_value=property_value
            )
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
        Search biological entities such as Gene, Protein, Disease, Chemical, Phenotype, Tissue, Anatomy,
        BiologicalProcess, MolecularFunction, CellularComponent, Pathway or Mutation by name or id

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
