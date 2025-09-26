"""
Hybrid Deployment Manager for On-Premises and Cloud Integration
"""
import asyncio
import logging
from typing import Dict, List, Optional
import json
import requests
from datetime import datetime
from config import settings

logger = logging.getLogger(__name__)

class HybridDeploymentManager:
    def __init__(self):
        self.on_prem_endpoint = settings.on_prem_endpoint
        self.cloud_endpoint = settings.cloud_endpoint
        self.deployment_mode = settings.deployment_mode
        self.sync_interval = 60  # seconds
        
    async def initialize_hybrid_deployment(self):
        """Initialize hybrid deployment configuration"""
        try:
            if self.deployment_mode == "hybrid":
                # Test connectivity to both endpoints
                on_prem_status = await self._test_endpoint_connectivity(self.on_prem_endpoint)
                cloud_status = await self._test_endpoint_connectivity(self.cloud_endpoint)
                
                logger.info(f"On-premises endpoint status: {on_prem_status}")
                logger.info(f"Cloud endpoint status: {cloud_status}")
                
                if on_prem_status and cloud_status:
                    # Start synchronization between endpoints
                    asyncio.create_task(self._sync_endpoints())
                    logger.info("Hybrid deployment initialized successfully")
                else:
                    logger.warning("Some endpoints are not accessible in hybrid mode")
            
            elif self.deployment_mode == "on-prem":
                on_prem_status = await self._test_endpoint_connectivity(self.on_prem_endpoint)
                logger.info(f"On-premises deployment status: {on_prem_status}")
            
            elif self.deployment_mode == "cloud":
                cloud_status = await self._test_endpoint_connectivity(self.cloud_endpoint)
                logger.info(f"Cloud deployment status: {cloud_status}")
                
        except Exception as e:
            logger.error(f"Error initializing hybrid deployment: {e}")
    
    async def _test_endpoint_connectivity(self, endpoint: str) -> bool:
        """Test connectivity to an endpoint"""
        try:
            if not endpoint:
                return False
                
            response = requests.get(f"{endpoint}/health", timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error testing endpoint {endpoint}: {e}")
            return False
    
    async def _sync_endpoints(self):
        """Synchronize data between on-premises and cloud endpoints"""
        while True:
            try:
                if self.deployment_mode == "hybrid":
                    # Sync data from on-prem to cloud
                    await self._sync_to_cloud()
                    
                    # Sync data from cloud to on-prem
                    await self._sync_from_cloud()
                
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"Error in endpoint synchronization: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def _sync_to_cloud(self):
        """Sync data from on-premises to cloud"""
        try:
            # Get recent data from on-premises
            on_prem_data = await self._get_recent_data(self.on_prem_endpoint)
            
            if on_prem_data:
                # Send to cloud
                await self._send_data_to_endpoint(self.cloud_endpoint, on_prem_data)
                logger.info(f"Synced {len(on_prem_data)} records to cloud")
                
        except Exception as e:
            logger.error(f"Error syncing to cloud: {e}")
    
    async def _sync_from_cloud(self):
        """Sync data from cloud to on-premises"""
        try:
            # Get recent data from cloud
            cloud_data = await self._get_recent_data(self.cloud_endpoint)
            
            if cloud_data:
                # Send to on-premises
                await self._send_data_to_endpoint(self.on_prem_endpoint, cloud_data)
                logger.info(f"Synced {len(cloud_data)} records from cloud")
                
        except Exception as e:
            logger.error(f"Error syncing from cloud: {e}")
    
    async def _get_recent_data(self, endpoint: str) -> List[Dict]:
        """Get recent data from an endpoint"""
        try:
            response = requests.get(
                f"{endpoint}/api/v1/sync/recent",
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("data", [])
            
        except Exception as e:
            logger.error(f"Error getting data from {endpoint}: {e}")
        
        return []
    
    async def _send_data_to_endpoint(self, endpoint: str, data: List[Dict]):
        """Send data to an endpoint"""
        try:
            response = requests.post(
                f"{endpoint}/api/v1/sync/receive",
                json={"data": data},
                timeout=30
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to send data to {endpoint}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending data to {endpoint}: {e}")
    
    async def route_request(self, request_type: str, data: Dict) -> Dict:
        """Route requests based on deployment mode and load"""
        try:
            if self.deployment_mode == "hybrid":
                # Route based on request type and current load
                endpoint = await self._select_optimal_endpoint(request_type, data)
            elif self.deployment_mode == "on-prem":
                endpoint = self.on_prem_endpoint
            elif self.deployment_mode == "cloud":
                endpoint = self.cloud_endpoint
            else:
                raise ValueError(f"Unknown deployment mode: {self.deployment_mode}")
            
            # Forward request to selected endpoint
            response = await self._forward_request(endpoint, request_type, data)
            return response
            
        except Exception as e:
            logger.error(f"Error routing request: {e}")
            return {"error": str(e)}
    
    async def _select_optimal_endpoint(self, request_type: str, data: Dict) -> str:
        """Select optimal endpoint for request routing"""
        try:
            # Get load information from both endpoints
            on_prem_load = await self._get_endpoint_load(self.on_prem_endpoint)
            cloud_load = await self._get_endpoint_load(self.cloud_endpoint)
            
            # Route based on request type and load
            if request_type in ["deepfake_detection", "video_kyc"]:
                # CPU-intensive tasks - prefer on-premises if available
                if on_prem_load < 0.8:
                    return self.on_prem_endpoint
                elif cloud_load < 0.8:
                    return self.cloud_endpoint
                else:
                    return self.on_prem_endpoint  # Default to on-premises
            
            elif request_type in ["identity_monitoring", "risk_assessment"]:
                # Data-intensive tasks - prefer cloud for scalability
                if cloud_load < 0.8:
                    return self.cloud_endpoint
                elif on_prem_load < 0.8:
                    return self.on_prem_endpoint
                else:
                    return self.cloud_endpoint  # Default to cloud
            
            else:
                # Default routing based on load
                if on_prem_load < cloud_load:
                    return self.on_prem_endpoint
                else:
                    return self.cloud_endpoint
                    
        except Exception as e:
            logger.error(f"Error selecting optimal endpoint: {e}")
            return self.on_prem_endpoint  # Fallback to on-premises
    
    async def _get_endpoint_load(self, endpoint: str) -> float:
        """Get current load of an endpoint"""
        try:
            response = requests.get(f"{endpoint}/api/v1/status/load", timeout=5)
            
            if response.status_code == 200:
                return response.json().get("load", 0.5)
            
        except Exception as e:
            logger.error(f"Error getting load from {endpoint}: {e}")
        
        return 0.5  # Default load
    
    async def _forward_request(self, endpoint: str, request_type: str, data: Dict) -> Dict:
        """Forward request to selected endpoint"""
        try:
            # Map request types to API endpoints
            endpoint_mapping = {
                "deepfake_detection": "/api/v1/deepfake/detect",
                "video_kyc": "/api/v1/kyc/video",
                "identity_monitoring": "/api/v1/monitoring/check",
                "risk_assessment": "/api/v1/risk/assess"
            }
            
            api_endpoint = endpoint_mapping.get(request_type, "/api/v1/process")
            
            response = requests.post(
                f"{endpoint}{api_endpoint}",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Request failed with status {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error forwarding request to {endpoint}: {e}")
            return {"error": str(e)}
    
    async def get_deployment_status(self) -> Dict:
        """Get current deployment status"""
        try:
            status = {
                "deployment_mode": self.deployment_mode,
                "endpoints": {
                    "on_prem": {
                        "url": self.on_prem_endpoint,
                        "status": await self._test_endpoint_connectivity(self.on_prem_endpoint)
                    },
                    "cloud": {
                        "url": self.cloud_endpoint,
                        "status": await self._test_endpoint_connectivity(self.cloud_endpoint)
                    }
                },
                "sync_status": {
                    "active": self.deployment_mode == "hybrid",
                    "interval": self.sync_interval
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting deployment status: {e}")
            return {"error": str(e)}
