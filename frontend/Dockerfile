# Sử dụng base image node:18-alpine
FROM node:18-alpine

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy package.json và cài đặt
COPY package.json .
RUN npm install

# Copy source code
COPY . .

# Mở port
EXPOSE 3000

# Lệnh chạy mặc định
CMD ["npm", "start"]
