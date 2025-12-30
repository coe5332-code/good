# BSK Training Optimization System - Streamlit Frontend

This is the Streamlit frontend for the BSK Training Optimization System, designed to be deployed on Streamlit Cloud.

## 🚀 Quick Start

### Prerequisites
- Backend API deployed at: `https://bsk-backend-uywi.onrender.com`
- Streamlit Cloud account (free tier available)

### Deployment to Streamlit Cloud

1. **Fork/Clone this repository**
   ```bash
   git clone https://github.com/coe5332-code/good.git
   cd good
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select this repository
   - Set main file path to: `frontend/app.py`
   - Click "Deploy"

3. **Configuration**
   - The app automatically uses the deployed backend at `https://bsk-backend-uywi.onrender.com`
   - No environment variables needed (backend URL is hardcoded)
   - If you need to override, set `API_BASE_URL` in Streamlit Cloud secrets

### Local Development

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**
   ```bash
   streamlit run frontend/app.py
   ```

3. **Access the app**
   - Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
.
├── frontend/
│   ├── app.py                    # Main Streamlit app
│   ├── pages/                    # Streamlit pages
│   │   ├── 01_BSK_Centers.py
│   │   ├── 02_Services.py
│   │   ├── 03_DEOs.py
│   │   ├── 04_Provisions.py
│   │   ├── 05_Service_Recommendation.py
│   │   └── 06_Underperforming_BSKs.py
│   └── utils.py                  # Utility functions
├── requirements.txt              # Python dependencies
├── .streamlit/
│   └── config.toml               # Streamlit configuration
└── README_STREAMLIT.md           # This file
```

## 🔧 Features

- **BSK Centers**: View and analyze BSK center data
- **Services**: Browse available services
- **DEOs**: Data Entry Operator management
- **Provisions**: Service provision tracking
- **Service Recommendations**: AI-powered BSK recommendations for new services
- **Underperforming BSKs**: Identify BSKs that need training

## 🌐 Backend API

The frontend connects to the backend API at:
- **Production**: `https://bsk-backend-uywi.onrender.com`
- **API Docs**: `https://bsk-backend-uywi.onrender.com/docs`

### Available Endpoints

- `GET /health` - Health check
- `GET /bsk/` - List all BSK centers
- `GET /services/` - List all services
- `GET /deo/` - List all DEOs
- `GET /provisions/` - List all provisions
- `GET /underperforming_bsks/` - Get underperforming BSKs
- `POST /refresh_data` - Refresh data cache

## 🐛 Troubleshooting

### Backend Connection Issues
- Ensure the backend is running and accessible
- Check backend health: `https://bsk-backend-uywi.onrender.com/health`
- Verify CORS is enabled on the backend

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

### Performance Issues
- The backend may take time to initialize on first request
- Large datasets may take time to load
- Use pagination controls in the sidebar

## 📝 Notes

- The app uses environment variable `API_BASE_URL` if set, otherwise defaults to the production backend
- All data is fetched from the backend API (no local database required)
- AI features require the backend to have proper ML models initialized

## 📄 License

This project is proprietary and confidential.

