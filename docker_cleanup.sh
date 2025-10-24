#!/bin/bash
# Docker cleanup script to free disk space

echo "Current Docker disk usage:"
docker system df

echo ""
echo "Cleaning up Docker resources..."

# Remove stopped containers
echo "Removing stopped containers..."
docker container prune -f

# Remove unused images
echo "Removing unused images..."
docker image prune -a -f

# Remove unused volumes (be careful!)
echo "Removing unused volumes..."
docker volume prune -f

# Remove build cache
echo "Removing build cache..."
docker builder prune -a -f

echo ""
echo "Cleanup complete! New disk usage:"
docker system df
