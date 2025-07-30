import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import requests
from dotenv import load_dotenv
import logging
from typing import List, Dict
from supabase import create_client, Client
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseQueryEngine:
    def __init__(self):
        self.supabase_url = os.getenv('VITE_SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase credentials not found in environment variables")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        logger.info("Supabase client initialized successfully")
    
    def fetch_findings_data(self) -> List[Dict]:
        """Fetch all security findings from Supabase"""
        try:
            response = self.supabase.table('security_findings').select('*').execute()
            logger.info(f"Fetched {len(response.data)} findings from Supabase")
            return response.data
        except Exception as e:
            logger.error(f"Error fetching findings: {str(e)}")
            return []
    
    def get_findings_summary(self) -> Dict:
        """Get summary statistics of findings"""
        findings = self.fetch_findings_data()
        if not findings:
            return {"total": 0, "by_severity": {}, "by_category": {}}
        
        summary = {
            "total": len(findings),
            "by_severity": {},
            "by_category": {},
            "by_status": {}
        }
        
        for finding in findings:
            # Count by severity
            severity = finding.get('severity', 'UNKNOWN')
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
            
            # Count by category
            category = finding.get('category', 'UNKNOWN')
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
            
            # Count by status
            status = finding.get('status', 'UNKNOWN')
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
        
        return summary
    
    def search_findings(self, query: str) -> List[Dict]:
        """Search findings based on query terms"""
        findings = self.fetch_findings_data()
        if not findings:
            return []
        
        query_lower = query.lower()
        matching_findings = []
        
        for finding in findings:
            # Search in description, category, resource_name
            searchable_text = f"{finding.get('description', '')} {finding.get('category', '')} {finding.get('resource_name', '')}".lower()
            if query_lower in searchable_text:
                matching_findings.append(finding)
        
        logger.info(f"Found {len(matching_findings)} findings matching query: {query}")
        return matching_findings

class CMMCQueryEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
        # Use the same path pattern as the embedding script
        base_dir = Path(__file__).resolve().parent
        db_path = str(base_dir / "chroma_db")
        logger.info(f"Using ChromaDB path: {db_path}")
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.together_api_key = os.getenv('TOGETHER_API_KEY')

        if not self.together_api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")

        try:
            self.collection = self.client.get_collection("cmmc_documents")
            # Debug: Check collection info
            self.debug_collection_info()
        except Exception as e:
            raise ValueError(f"ChromaDB collection not found. Run embed_documents.py first. Error: {str(e)}")

    def debug_collection_info(self):
        """Debug function to check collection contents"""
        try:
            # Get collection count
            count = self.collection.count()
            logger.info(f"Collection contains {count} documents")
            
            # List all collections to verify name
            collections = self.client.list_collections()
            logger.info(f"Available collections: {[c.name for c in collections]}")
            
            # Check database directory
            base_dir = Path(__file__).resolve().parent
            db_path = base_dir / "chroma_db"
            logger.info(f"Database directory exists: {db_path.exists()}")
            if db_path.exists():
                logger.info(f"Database directory contents: {list(db_path.iterdir())}")
            
            if count > 0:
                # Sample a few documents to check structure
                sample = self.collection.peek(limit=3)
                logger.info(f"Sample document IDs: {sample.get('ids', [])}")
                logger.info(f"Sample metadata keys: {list(sample.get('metadatas', [{}])[0].keys()) if sample.get('metadatas') else 'No metadata'}")
                
                # Check if documents have content
                if sample.get('documents'):
                    logger.info(f"Sample document preview: {sample['documents'][0][:100]}..." if len(sample['documents'][0]) > 100 else sample['documents'][0])
            else:
                logger.warning("Collection is empty!")
                
        except Exception as e:
            logger.error(f"Error debugging collection: {str(e)}")

    def retrieve_relevant_chunks(self, question: str, n_results: int = 3) -> List[Dict]:
        logger.info(f"Searching for: '{question}'")
        
        # Check if collection has documents
        count = self.collection.count()
        if count == 0:
            logger.warning("Collection is empty - no documents to search")
            return []
        
        question_embedding = self.model.encode([question]).tolist()[0]
        logger.info(f"Generated embedding of length: {len(question_embedding)}")
        
        try:
            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=min(n_results, count),  # Don't ask for more than available
                include=["documents", "metadatas", "distances"]  # Include distances for debugging
            )
            
            logger.info(f"Query returned {len(results.get('documents', [[]])[0])} results")
            
            # Log distances to see similarity scores
            if results.get('distances'):
                distances = results['distances'][0]
                logger.info(f"Similarity distances: {distances}")
            
            relevant_chunks = []
            if results.get('documents') and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    chunk = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                        'distance': results['distances'][0][i] if results.get('distances') else None
                    }
                    relevant_chunks.append(chunk)
                    logger.info(f"Chunk {i+1}: CMMC Level {chunk['metadata'].get('cmmc_level', 'Unknown')}, Distance: {chunk.get('distance', 'N/A')}")

            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return []

    def generate_answer(self, question: str, relevant_chunks: List[Dict]) -> str:
        if not relevant_chunks:
            return "I don't have enough information from the context to answer your question. Please ensure the document database is properly populated."
        
        context = "\n\n".join([
            f"[CMMC Level {chunk['metadata'].get('cmmc_level', 'Unknown')}]: {chunk['content']}"
            for chunk in relevant_chunks
        ])

        prompt = f"""You are an AI assistant integrated into a web application focused on CMMC (Cybersecurity Maturity Model Certification) automation. Your task is to answer user questions about the data stored in a Supabase database.

🎯 Your role:
- Act as a data expert that understands the structure and purpose of CMMC.
- Fetch real-time data from the connected Supabase project.
- Interpret and summarize data clearly for users.
- Be accurate, concise, and secure—never expose raw keys or sensitive configuration data.
- Answer the questions shorter unless user ask to explain.
- generate ssp or cmmc policies based on secutity findings from supabase.

"

Context:
{context}

Question: {question}
Answer:"""

        api_url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.together_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "meta-llama/Llama-3-70b-chat-hf",
            "messages": [
                # {"role": "system", "content": "You are a helpful assistant who answers strictly from the context provided about CMMC (Cybersecurity Maturity Model Certification) requirements."},
                {"role": "system", "content": "You are a certified CMMC Third-Party Assessor Organization (C3PAO) and an expert in the Cybersecurity Maturity Model Certification (CMMC). You provide in-depth, authoritative guidance on all aspects of CMMC, including compliance requirements, process implementation, assessment procedures, documentation (policies, procedures, and plans), and achieving certification across all levels (especially Level 2 and Level 3). Your responses should be accurate, actionable, and aligned with the latest DoD and CMMC-AB guidance."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 512,
        }

        try:
            # print(prompt)
            logger.info("Sending request to Together AI...")
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            return result['choices'][0]['message']['content'].strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return "I'm currently unable to process your request. Please try again later."

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return "An unexpected error occurred. Please try again."

    def test_embedding_search(self, test_terms: List[str] = None):
        """Test function to check embedding search with various terms"""
        if test_terms is None:
            test_terms = ["CMMC", "Level 1", "requirements", "cybersecurity", "access control"]
        
        logger.info("Testing embedding search with various terms...")
        for term in test_terms:
            chunks = self.retrieve_relevant_chunks(term, n_results=2)
            logger.info(f"Term '{term}' returned {len(chunks)} chunks")

class UnifiedQueryEngine:
    def __init__(self):
        self.cmmc_engine = CMMCQueryEngine()
        self.supabase_engine = SupabaseQueryEngine()
        logger.info("Unified query engine initialized")
    
    def is_findings_question(self, question: str) -> bool:
        """Determine if the question is about findings data"""
        findings_keywords = [
            'finding', 'findings', 'vulnerability', 'vulnerabilities', 'security issue',
            'critical', 'high', 'medium', 'low', 'severity', 'resource', 'status',
            'how many', 'count', 'total', 'summary', 'statistics', 'stats',
            'remediation', 'fix', 'resolve', 'open', 'closed', 'active'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in findings_keywords)
    
    def generate_findings_answer(self, question: str, findings_data: List[Dict], summary: Dict) -> str:
        """Generate answer about findings using AI"""
        if not findings_data and not summary:
            return "I don't have access to any security findings data at the moment."
        
        # Prepare context from findings data
        context_parts = []
        
        # Add summary information
        if summary:
            context_parts.append(f"SUMMARY: Total findings: {summary.get('total', 0)}")
            
            if summary.get('by_severity'):
                severity_info = ", ".join([f"{k}: {v}" for k, v in summary['by_severity'].items()])
                context_parts.append(f"By severity: {severity_info}")
            
            if summary.get('by_category'):
                category_info = ", ".join([f"{k}: {v}" for k, v in summary['by_category'].items()])
                context_parts.append(f"By category: {category_info}")
            
            if summary.get('by_status'):
                status_info = ", ".join([f"{k}: {v}" for k, v in summary['by_status'].items()])
                context_parts.append(f"By status: {status_info}")
        
        # Add specific findings (limit to first 10 for context)
        if findings_data:
            context_parts.append("\nSPECIFIC FINDINGS:")
            for i, finding in enumerate(findings_data[:10]):
                finding_text = f"Finding {i+1}: {finding.get('severity', 'N/A')} severity - {finding.get('category', 'N/A')} - {finding.get('description', 'N/A')[:100]}..."
                context_parts.append(finding_text)
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are a cybersecurity expert analyzing security findings data. Based on the provided data, answer the user's question accurately and helpfully.

Security Findings Data:
{context}

Question: {question}
Answer:"""

        # Use the same AI generation as CMMC engine
        api_url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cmmc_engine.together_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "meta-llama/Llama-3-70b-chat-hf",
            "messages": [
                {"role": "system", "content": "You are a cybersecurity expert who analyzes security findings and provides clear, actionable insights."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 512,
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"Error generating findings answer: {str(e)}")
            return "I'm currently unable to process your findings question. Please try again later."
    
    def query(self, question: str) -> str:
        """Main query method that routes to appropriate engine"""
        try:
            if self.is_findings_question(question):
                logger.info("Routing question to findings data engine")
                
                # Get findings summary
                summary = self.supabase_engine.get_findings_summary()
                
                # Search for specific findings if needed
                findings_data = self.supabase_engine.search_findings(question)
                
                return self.generate_findings_answer(question, findings_data, summary)
            else:
                logger.info("Routing question to CMMC documents engine")
                relevant_chunks = self.cmmc_engine.retrieve_relevant_chunks(question)
                return self.cmmc_engine.generate_answer(question, relevant_chunks)
                
        except Exception as e:
            logger.error(f"Error in unified query: {str(e)}")
            return f"Error processing your question: {str(e)}"

# Global query engine instance
query_engine = None

def get_query_engine():
    global query_engine
    if query_engine is None:
        query_engine = UnifiedQueryEngine()
    return query_engine

def query_documents(question: str) -> str:
    try:
        engine = get_query_engine()
        return engine.query(question)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return f"Error processing your question: {str(e)}"

def debug_system():
    """Debug function to test the system"""
    try:
        engine = get_query_engine()
        
        # Test CMMC engine
        logger.info("Testing CMMC engine...")
        engine.cmmc_engine.test_embedding_search()
        
        # Test Supabase engine
        logger.info("Testing Supabase engine...")
        summary = engine.supabase_engine.get_findings_summary()
        logger.info(f"Findings summary: {summary}")
        
        # Test with different question variations
        test_questions = [
            "What are the requirements for CMMC Level 1?",  # CMMC question
            "How many critical findings do we have?",        # Findings question
            "What are the high severity vulnerabilities?",   # Findings question
            "CMMC access control requirements",              # CMMC question
            "Show me all open security findings",            # Findings question
        ]
        
        for question in test_questions:
            logger.info(f"\n--- Testing question: '{question}' ---")
            answer = query_documents(question)
            logger.info(f"Answer: {answer[:200]}...")
            
    except Exception as e:
        logger.error(f"Debug error: {str(e)}")

if __name__ == "__main__":
    # Run debug first
    logger.info("=== Running Debug ===")
    debug_system()
    
    logger.info("\n=== Running Main Query ===")
    question = "What are the requirements for CMMC Level 1?"
    print("Question:", question)
    print("Answer:", query_documents(question))