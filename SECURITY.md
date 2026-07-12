# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it responsibly.

## Security Measures

### API Key Management
- All API keys are stored as environment variables
- `.env` files are excluded from version control via `.gitignore`
- Streamlit Cloud secrets are used for deployment

### SQL Injection Prevention
- All user inputs are parameterized in SQL queries
- Dangerous operations (DROP, DELETE, UPDATE, INSERT, ALTER, CREATE) are blocked at the application level
- Natural language to SQL conversion includes validation layer

### Input Validation
- All user inputs are sanitized before processing
- File uploads are type-restricted (audio formats only)
- Text inputs have length limits enforced

### Data Privacy
- No personal user data is collected or stored
- All data is synthetic/demo data
- No authentication tokens are logged
- Audio files are processed in memory and immediately deleted

### Dependencies
- All dependencies are pinned to minimum versions
- Regular updates recommended for security patches
