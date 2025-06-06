# 🔋 AdEva-ausgrid Energy Consumption Analysis Platform

## 🎯 Overview
AdEva-ausgrid is a full-stack web application for analyzing energy consumption data from Ausgrid. The platform provides tools for monitoring, analyzing, and predicting energy consumption patterns across different consumers and regions.

## ✨ Features
- 🔐 User Authentication System
- 👥 Consumer Management
- 📊 Energy Consumption Analytics
- 🌓 Dark/Light Theme Support
- 📱 Interactive Dashboard
- 🔍 Consumer Data Filtering
- 📥 Data Import Capabilities

## 🛠️ Tech Stack
### 🎨 Frontend
- ⚛️ React.js
- 🎭 Material-UI
- 🌈 Tailwind CSS
- 🦸 Heroicons
- 📦 Context API for state management

### 🔧 Backend
- 🐍 Django
- 🌐 Django REST Framework
- 🍃 MongoDB with MongoEngine
- 🔑 JWT Authentication

## 📁 Project Structure
```
AdEva-ausgrid/
├── app/
│   ├── frontend/           # React frontend application
│   │   ├── public/        # Static files
│   │   └── src/          # Source code
│   └── backend/           # Django backend application
│       └── project_root/  # Django project root
├── docker-compose.yml     # Docker composition file
└── README.md             # This file
```
### 🌐 Access Points
The application will be available at:

- 🖥️ Frontend: http://localhost:3000
- ⚡ Backend API: http://localhost:8000

### 💻 Development Setup Frontend
```bash
cd app/frontend
npm install
npm start
```
 Backend
```bash
cd app/backend/project_root
pip install -r requirements.txt
python manage.py runserver
 ```


### 🔌 API Endpoints
- 🔑 /api/token/ - JWT token generation
- 🔄 /api/token/refresh/ - JWT token refresh
- 💓 /health/ - Health check endpoint
- 👥 /consumers/ - Consumer management
- 📊 /records/ - Energy records
- 🔮 /predict/ - Consumption prediction
- 📤 /upload/ - Bulk data upload

## 🌟 Features Details
1. 🔐 Authentication System
   
   - JWT-based authentication
   - User registration and login
   - Password encryption
2. 👥 Consumer Management
   
   - Consumer listing
   - Consumer filtering
   - Consumption tracking
3. 📊 Data Visualization
   
   - Interactive dashboard
   - Statistical analysis
   - Consumption patterns
4. 🎨 Theme Support
   
   - Dark/Light mode toggle
   - Responsive design
   - Modern UI components

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## 📄 License
This project is licensed under the MIT License.