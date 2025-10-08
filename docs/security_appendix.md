# Maritime ACPS Security Appendix

## Security Policy Statement

**The Maritime Autonomous Convoy Protection System (ACPS) is designed as a DEFENSIVE-ONLY system. All engagement decisions require explicit human authorization. The system provides threat detection and route recommendations only - no automated kinetic responses are permitted.**

## Human-in-the-Loop Requirements

### Critical Safety Controls

1. **No Automated Weapons Engagement**
   - System cannot directly control weapons systems
   - All kinetic responses require human operator approval
   - Clear separation between detection and engagement systems

2. **Human Authorization Gates**
   - Route changes require operator approval
   - High-confidence threats trigger human review
   - Emergency procedures include human override capabilities

3. **Audit and Accountability**
   - All decisions logged with operator identification
   - Complete audit trail for threat detections
   - Regular review of automated recommendations

## AWS Security Best Practices

### Identity and Access Management

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::maritime-acps-data-*/*"
    }
  ]
}
```

### Network Security

- **VPC Configuration**: Private subnets for sensitive workloads
- **Security Groups**: Minimal port exposure (HTTPS, MQTT only)
- **NACLs**: Additional network-level filtering
- **VPC Endpoints**: Private connectivity to AWS services

### Data Protection

- **Encryption at Rest**: KMS keys for S3, DynamoDB, EBS
- **Encryption in Transit**: TLS 1.2+ for all communications
- **Key Management**: Separate keys per environment
- **Data Classification**: Clear labeling of sensitive data

## IoT Security

### Device Authentication

```bash
# Certificate-based authentication
aws iot create-keys-and-certificate \
  --set-as-active \
  --certificate-pem-outfile device.cert.pem \
  --public-key-outfile device.public.key \
  --private-key-outfile device.private.key
```

### IoT Policies

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "iot:Connect",
        "iot:Publish"
      ],
      "Resource": [
        "arn:aws:iot:region:account:client/maritime-*",
        "arn:aws:iot:region:account:topic/maritime/detections"
      ]
    }
  ]
}
```

### Greengrass Security

- **Component Isolation**: Separate containers for inference
- **Local Secrets**: Secure credential storage
- **OTA Security**: Signed component updates
- **Network Isolation**: Local-only inference when offline

## ML Model Security

### Training Security

- **Data Validation**: Schema validation for training data
- **Model Versioning**: Immutable model artifacts
- **Experiment Tracking**: Complete lineage of model development
- **Access Controls**: Restricted access to training jobs

### Inference Security

- **Model Integrity**: Cryptographic signatures for models
- **Input Validation**: Sanitization of sensor inputs
- **Output Filtering**: Confidence thresholds and anomaly detection
- **Monitoring**: Real-time model performance tracking

## Application Security

### Frontend Security

```javascript
// Cognito configuration with MFA
const authConfig = {
  region: 'us-east-1',
  userPoolId: 'us-east-1_XXXXXXXXX',
  userPoolWebClientId: 'XXXXXXXXX',
  mandatorySignIn: true,
  authenticationFlowType: 'USER_SRP_AUTH'
};
```

### API Security

- **Authentication**: Cognito JWT tokens
- **Authorization**: Role-based access control
- **Rate Limiting**: API Gateway throttling
- **Input Validation**: Schema validation for all inputs

### WebSocket Security

- **Connection Authentication**: Token-based auth
- **Message Validation**: Structured message formats
- **Connection Limits**: Per-user connection limits
- **Audit Logging**: All WebSocket events logged

## Operational Security

### Monitoring and Alerting

```yaml
# CloudWatch Alarm for suspicious activity
SuspiciousActivityAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    AlarmName: Maritime-SuspiciousActivity
    MetricName: UnauthorizedAccess
    Threshold: 1
    ComparisonOperator: GreaterThanOrEqualToThreshold
    AlarmActions:
      - !Ref SecurityNotificationTopic
```

### Incident Response

1. **Detection**: Automated monitoring and alerting
2. **Assessment**: Security team evaluation
3. **Containment**: Isolation of affected systems
4. **Recovery**: Restoration of normal operations
5. **Lessons Learned**: Post-incident review

### Backup and Recovery

- **Data Backup**: Automated S3 cross-region replication
- **Configuration Backup**: Infrastructure as Code in version control
- **Recovery Testing**: Regular disaster recovery drills
- **RTO/RPO**: 4-hour recovery time, 1-hour data loss maximum

## Compliance Framework

### Security Standards

- **NIST Cybersecurity Framework**: Core security controls
- **ISO 27001**: Information security management
- **SOC 2 Type II**: Service organization controls
- **FedRAMP**: Federal security requirements (if applicable)

### Data Governance

- **Data Classification**: Public, Internal, Confidential, Restricted
- **Retention Policies**: Automated data lifecycle management
- **Privacy Controls**: No PII in training datasets
- **Cross-Border**: Data residency requirements

## Threat Model

### Attack Vectors

1. **Edge Device Compromise**
   - Mitigation: Certificate-based auth, local monitoring
   
2. **Cloud Service Exploitation**
   - Mitigation: Least privilege IAM, network segmentation
   
3. **ML Model Poisoning**
   - Mitigation: Data validation, model versioning
   
4. **Dashboard Compromise**
   - Mitigation: MFA, session management, HTTPS

### Risk Assessment

| Threat | Likelihood | Impact | Risk Level | Mitigation |
|--------|------------|--------|------------|------------|
| Device Tampering | Medium | High | High | Physical security, attestation |
| Data Exfiltration | Low | High | Medium | Encryption, access controls |
| Model Evasion | Medium | Medium | Medium | Ensemble models, monitoring |
| DoS Attack | High | Low | Medium | Rate limiting, auto-scaling |

## Security Testing

### Automated Testing

```bash
# Security scan in CI/CD
bandit -r . -f json -o security-report.json
safety check --json --output safety-report.json
```

### Penetration Testing

- **Quarterly**: External penetration testing
- **Continuous**: Automated vulnerability scanning
- **Red Team**: Annual adversarial testing
- **Bug Bounty**: Responsible disclosure program

## Emergency Procedures

### Security Incident Response

1. **Immediate Actions**
   - Isolate affected systems
   - Preserve evidence
   - Notify security team

2. **Communication Plan**
   - Internal stakeholders
   - External authorities (if required)
   - Public disclosure (if applicable)

3. **Recovery Steps**
   - System restoration
   - Security hardening
   - Monitoring enhancement

### System Shutdown

```bash
# Emergency system shutdown
aws iot update-thing --thing-name convoy-edge-01 --attribute-payload '{"attributes":{"emergency_stop":"true"}}'
aws ecs update-service --cluster maritime-cluster --service planner --desired-count 0
```

## Security Contacts

- **Security Team**: security@maritime-acps.com
- **Incident Response**: incident@maritime-acps.com
- **Vulnerability Reports**: security-reports@maritime-acps.com

## Legal and Ethical Considerations

### Use Restrictions

- **Defensive Operations Only**: No offensive capabilities
- **Human Oversight**: Required for all critical decisions
- **International Waters**: Compliance with maritime law
- **Export Controls**: ITAR/EAR compliance for technology transfer

### Ethical AI Principles

- **Transparency**: Explainable AI decisions
- **Accountability**: Human responsibility for outcomes
- **Fairness**: Unbiased threat detection
- **Privacy**: Minimal data collection and retention