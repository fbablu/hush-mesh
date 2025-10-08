import json
import boto3
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime')
sagemaker_client = boto3.client('sagemaker')

def get_active_endpoint():
    """Get the most recent active maritime endpoint"""
    try:
        # Check environment variable first
        endpoint_name = os.environ.get('SAGEMAKER_ENDPOINT_NAME')
        if endpoint_name:
            return endpoint_name
            
        # List endpoints and find maritime ones
        response = sagemaker_client.list_endpoints(
            StatusEquals='InService',
            NameContains='maritime-acps-endpoint'
        )
        
        if response['Endpoints']:
            # Return the most recent one
            endpoints = sorted(response['Endpoints'], key=lambda x: x['CreationTime'], reverse=True)
            return endpoints[0]['EndpointName']
        
        # Fallback to hardcoded name
        return 'maritime-acps-endpoint-1759946037'
        
    except Exception as e:
        logger.error(f"Error getting endpoint: {e}")
        return 'maritime-acps-endpoint-1759946037'

def lambda_handler(event, context):
    """Lambda function to call SageMaker endpoint"""
    
    try:
        # Parse request
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event
        
        # Get active endpoint
        endpoint_name = get_active_endpoint()
        logger.info(f"Using endpoint: {endpoint_name}")
        
        # Call SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(body)
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        logger.error(f"Error calling ML endpoint: {str(e)}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'endpoint': get_active_endpoint(),
                'message': 'SageMaker endpoint error - check endpoint status'
            })
        }

def handler(event, context):
    """Alternative handler name"""
    return lambda_handler(event, context)