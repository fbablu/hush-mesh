#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { MaritimeAcpsStack } from '../lib/maritime-acps-stack';

const app = new cdk.App();
new MaritimeAcpsStack(app, 'MaritimeAcpsStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
});