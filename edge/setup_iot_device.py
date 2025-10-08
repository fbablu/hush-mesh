#!/usr/bin/env python3
import boto3
import json
import argparse
import os
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes

def create_iot_device(device_name, region='us-east-1'):
    """Create IoT Thing, certificates, and policies"""
    
    iot_client = boto3.client('iot', region_name=region)
    
    # Create IoT Thing
    try:
        thing_response = iot_client.create_thing(thingName=device_name)
        print(f"Created IoT Thing: {device_name}")
    except iot_client.exceptions.ResourceAlreadyExistsException:
        print(f"IoT Thing {device_name} already exists")
    
    # Create certificate
    cert_response = iot_client.create_keys_and_certificate(setAsActive=True)
    
    certificate_arn = cert_response['certificateArn']
    certificate_id = cert_response['certificateId']
    certificate_pem = cert_response['certificatePem']
    private_key = cert_response['keyPair']['PrivateKey']
    public_key = cert_response['keyPair']['PublicKey']
    
    # Save certificates to files
    os.makedirs('certs', exist_ok=True)
    
    with open(f'certs/{device_name}-certificate.pem.crt', 'w') as f:
        f.write(certificate_pem)
    
    with open(f'certs/{device_name}-private.pem.key', 'w') as f:
        f.write(private_key)
    
    with open(f'certs/{device_name}-public.pem.key', 'w') as f:
        f.write(public_key)
    
    # Create IoT policy
    policy_name = f'{device_name}-policy'
    policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "iot:Connect",
                    "iot:Publish",
                    "iot:Subscribe",
                    "iot:Receive"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "greengrass:*"
                ],
                "Resource": "*"
            }
        ]
    }
    
    try:
        iot_client.create_policy(
            policyName=policy_name,
            policyDocument=json.dumps(policy_document)
        )
        print(f"Created IoT Policy: {policy_name}")
    except iot_client.exceptions.ResourceAlreadyExistsException:
        print(f"IoT Policy {policy_name} already exists")
    
    # Attach policy to certificate
    iot_client.attach_policy(
        policyName=policy_name,
        target=certificate_arn
    )
    
    # Attach certificate to thing
    iot_client.attach_thing_principal(
        thingName=device_name,
        principal=certificate_arn
    )
    
    # Get IoT endpoint
    endpoint_response = iot_client.describe_endpoint(endpointType='iot:Data-ATS')
    iot_endpoint = endpoint_response['endpointAddress']
    
    # Save device configuration
    device_config = {
        'device_name': device_name,
        'certificate_arn': certificate_arn,
        'certificate_id': certificate_id,
        'iot_endpoint': iot_endpoint,
        'region': region
    }
    
    with open(f'certs/{device_name}-config.json', 'w') as f:
        json.dump(device_config, f, indent=2)
    
    print(f"Device configuration saved to certs/{device_name}-config.json")
    print(f"IoT Endpoint: {iot_endpoint}")
    
    return device_config

def main():
    parser = argparse.ArgumentParser(description='Setup IoT device for maritime edge deployment')
    parser.add_argument('--device-name', required=True, help='Name of the IoT device')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    
    args = parser.parse_args()
    
    config = create_iot_device(args.device_name, args.region)
    print("IoT device setup complete!")

if __name__ == '__main__':
    main()