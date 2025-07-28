# Security Policy

## Supported Versions

We actively support the following versions of the Prompt Engineering Toolkit with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.0.x   | :x:                |

## Reporting a Vulnerability

We take the security of the Prompt Engineering Toolkit seriously. If you discover a security vulnerability, please follow these guidelines:

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing: **security@prompt-engineering-toolkit.org**

Include the following information in your report:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact of the vulnerability
- Any suggested fixes or mitigations
- Your contact information (optional, but helpful for follow-up)

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours.

2. **Initial Assessment**: We will provide an initial assessment of the report within 5 business days, including:
   - Whether we consider it a security issue
   - Severity level (Critical, High, Medium, Low)
   - Expected timeline for resolution

3. **Resolution**: We aim to resolve security issues according to their severity:
   - **Critical**: 1-7 days
   - **High**: 7-14 days  
   - **Medium**: 14-30 days
   - **Low**: 30-90 days

4. **Disclosure**: After the issue is resolved:
   - We will release a security update
   - We will publish a security advisory
   - We will credit you in our security acknowledgments (if desired)

### Security Best Practices

When using the Prompt Engineering Toolkit, please follow these security best practices:

#### API Key Management
- **Never commit API keys to version control**
- Use environment variables or secure key management systems
- Rotate API keys regularly
- Use separate keys for development, staging, and production

#### Web Interface Security
- Run the web interface behind a reverse proxy (nginx, Apache)
- Use HTTPS in production environments
- Implement proper authentication and authorization
- Keep the application and dependencies updated

#### Docker Security
- Use the official Docker images from trusted registries
- Scan container images for vulnerabilities
- Run containers with non-root users
- Use secrets management for sensitive configuration

#### Network Security
- Restrict network access to necessary ports only
- Use firewalls and security groups appropriately
- Monitor network traffic for suspicious activity
- Implement rate limiting for API endpoints

### Known Security Considerations

#### LLM API Interactions
- The toolkit sends prompts to external LLM APIs
- Ensure compliance with your organization's data governance policies
- Be aware that test prompts may contain sensitive or harmful content
- Monitor API usage and costs to prevent abuse

#### Prompt Data Sensitivity
- Red team prompts may contain sensitive or harmful content
- Implement appropriate access controls for prompt databases
- Consider data residency and compliance requirements
- Regularly audit and review prompt collections

#### Web Interface Risks
- The web interface processes user input and displays LLM responses
- Implement proper input validation and output encoding
- Use Content Security Policy (CSP) headers
- Regular security testing and vulnerability assessments

### Security Updates

Security updates will be released as patch versions (e.g., 2.0.1, 2.0.2) and will include:
- Detailed security advisory
- Migration instructions if needed
- Acknowledgment of security researchers (with permission)

Subscribe to our security notifications:
- Watch the GitHub repository for security advisories
- Follow our release notes for security-related changes

### Security Acknowledgments

We would like to thank the following security researchers for responsibly disclosing vulnerabilities:

*No security issues have been reported yet.*

### Contact Information

For security-related questions or concerns:
- **Email**: security@prompt-engineering-toolkit.org
- **PGP Key**: Available upon request
- **Response Time**: Within 48 hours

For general questions about security practices:
- Open a discussion in [GitHub Discussions](https://github.com/WolfgangDremmler/prompt-engineering-toolkit/discussions)
- Tag your question with "security"

### Legal

This security policy is subject to our [Terms of Service] and [Privacy Policy]. We reserve the right to modify this policy at any time. The most current version will always be available in this repository.

---

**Last Updated**: December 28, 2024
**Policy Version**: 1.0