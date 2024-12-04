FROM node:18-alpine AS build

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock)
COPY package*.json ./
# If using yarn, uncomment the next line and comment out npm install
# COPY yarn.lock ./

# Install dependencies
RUN npm install
# If using yarn, replace the above with:
# RUN yarn install

# Copy the rest of the application code
COPY . .

# Build the application
RUN npm run build
# If using yarn, replace with:
# RUN yarn build

# Production stage
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Install serve to run the application
RUN npm install -g serve

# Copy built assets from the build stage
COPY --from=build /app/build ./build
COPY --from=build /app/package.json ./

# Expose port (typically 3000 for React apps)
EXPOSE 3000

# Command to run the application
CMD ["serve", "-s", "build", "-l", "3000"]