import requests
from app.core.config import settings
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

class ICDService:
    def __init__(self):
        self.api_key = settings.ICD_API_KEY
        self.base_url = settings.ICD_API_URL
        self.headers = {
            "API-Version": "v2",
            "Accept-Language": "en",
            "Authorization": f"Bearer {self.api_key}"
        }

    async def search_diagnosis(self, query: str) -> Optional[List[Dict]]:
        """
        Search for ICD codes based on diagnosis text
        """
        try:
            response = requests.get(
                f"{self.base_url}/search",
                headers=self.headers,
                params={
                    "q": query,
                    "flatResults": "true",
                    "propertiesToBeSearched": "Title,Exclusion,FullySpecifiedName",
                    "useFlexisearch": "true"
                }
            )
            response.raise_for_status()
            return response.json().get("destinationEntities", [])
        except Exception as e:
            logger.error(f"Error searching ICD codes: {str(e)}")
            return None

    async def get_icd_code(self, diagnosis: str) -> Optional[str]:
        """
        Get the most relevant ICD code for a diagnosis
        """
        try:
            results = await self.search_diagnosis(diagnosis)
            if not results:
                return None
            
            # Get the first (most relevant) result
            first_result = results[0]
            return first_result.get("id")
        except Exception as e:
            logger.error(f"Error getting ICD code: {str(e)}")
            return None

icd_service = ICDService() 