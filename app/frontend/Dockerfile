# Dockerfile for React (frontend)
FROM node:20.14

WORKDIR /frontend

# Copy package.json and package-lock.json for installing dependencies
COPY package*.json ./

RUN npm install

# Copy the source code into the container
COPY . .

EXPOSE 3000

CMD ["npm", "start"]
