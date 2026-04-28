# Build Stage
FROM rust:1.76-slim-bullseye AS builder

# Install necessary build dependencies (C compiler for SQLite)
RUN apt-get update && apt-get install -y pkg-config libssl-dev gcc libc6-dev

WORKDIR /app
COPY . .

# Build the server binary
RUN cargo build --release --bin server

# Runtime Stage
FROM debian:bullseye-slim

WORKDIR /app

# Install runtime dependencies for SQLite
RUN apt-get update && apt-get install -y libsqlite3-0 ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary from the builder stage
COPY --from=builder /app/target/release/server /app/server
COPY --from=builder /app/docs /app/docs

# Expose the port axum uses
EXPOSE 3000

# Set environment variables for Google Cloud Run
ENV PORT=3000
ENV HOST=0.0.0.0

# Run the server
CMD ["/app/server"]
