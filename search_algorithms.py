import numpy as np
from collections import deque
import time
import heapq

def load_embeddings(filepath):
    embeddings = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=float)
            embeddings[word] = vector
            
    return embeddings

embeddings = load_embeddings("glove.100d.20000.txt")

def calculate_similarity(vec1, vec2):
    cos_sim= np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    distance = 1- cos_sim
    return distance

def get_k_neighbors(current_word, embeddings, k):
    current_vec = embeddings[current_word]
    
    similarities = []
    
    for word, vec in embeddings.items():
        if word == current_word:
            continue
            
        sim = calculate_similarity(current_vec, vec)
        similarities.append((word, sim))
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return only words
    neighbors = [word for word, _ in similarities[:k]]
    
    return neighbors

def get_path_cost(word1, word2, embeddings):
    vec1 = embeddings[word1]
    vec2 = embeddings[word2]
    
    similarity = calculate_similarity(vec1, vec2)
    return 1 - similarity

def calculate_heuristic(current_word, goal_word, embeddings):
    vec1 = embeddings[current_word]
    vec2 = embeddings[goal_word]
    
    similarity = calculate_similarity(vec1, vec2)
    return 1 - similarity

#searches implementations uniinformed
from collections import deque
import time

def bfs(start_word, goal_word, k, embeddings):
    start_time = time.time()
    
    queue = deque()
    queue.append((start_word, [start_word]))
    
    visited = set()
    visited.add(start_word)
    
    nodes_expanded = 0
    
    while queue:
        current_word, path = queue.popleft()
        nodes_expanded += 1
        
        if current_word == goal_word:
            runtime = time.time() - start_time
            return path, len(path), nodes_expanded, runtime
        
        neighbors = get_k_neighbors(current_word, embeddings, k)
        
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None, 0, nodes_expanded, time.time() - start_time

def dfs(start_word, goal_word, k, depth_limit, embeddings):
    start_time = time.time()
    
    stack = [(start_word, [start_word])]
    visited = set()
    nodes_expanded = 0
    
    while stack:
        current_word, path = stack.pop()
        nodes_expanded += 1
        
        if current_word == goal_word:
            runtime = time.time() - start_time
            return path, len(path), nodes_expanded, runtime
        
        if len(path) >= depth_limit:
            continue
        
        if current_word not in visited:
            visited.add(current_word)
            
            neighbors = get_k_neighbors(current_word, embeddings, k)
            
            for neighbor in neighbors:
                stack.append((neighbor, path + [neighbor]))
    
    return None, 0, nodes_expanded, time.time() - start_time

import heapq

def ucs(start_word, goal_word, k, embeddings):
    start_time = time.time()
    
    pq = []
    heapq.heappush(pq, (0, start_word, [start_word]))
    
    visited = {}
    nodes_expanded = 0
    
    while pq:
        cost, current_word, path = heapq.heappop(pq)
        nodes_expanded += 1
        
        if current_word == goal_word:
            runtime = time.time() - start_time
            return path, len(path), nodes_expanded, runtime
        
        if current_word in visited and visited[current_word] <= cost:
            continue
        
        visited[current_word] = cost
        
        neighbors = get_k_neighbors(current_word, embeddings, k)
        
        for neighbor in neighbors:
            step_cost = get_path_cost(current_word, neighbor, embeddings)
            heapq.heappush(pq, (cost + step_cost, neighbor, path + [neighbor]))
    
    return None, 0, nodes_expanded, time.time() - start_time

#informed searcges
def greedy_best_first(start_word, goal_word, k, embeddings):
    start_time = time.time()
    
    pq = []
    heapq.heappush(pq, (0, start_word, [start_word]))
    
    visited = set()
    nodes_expanded = 0
    
    while pq:
        _, current_word, path = heapq.heappop(pq)
        nodes_expanded += 1
        
        if current_word == goal_word:
            runtime = time.time() - start_time
            return path, len(path), nodes_expanded, runtime
        
        if current_word in visited:
            continue
        
        visited.add(current_word)
        
        neighbors = get_k_neighbors(current_word, embeddings, k)
        
        for neighbor in neighbors:
            h = calculate_heuristic(neighbor, goal_word, embeddings)
            heapq.heappush(pq, (h, neighbor, path + [neighbor]))
    
    return None, 0, nodes_expanded, time.time() - start_time

def a_star(start_word, goal_word, k, embeddings):
    start_time = time.time()
    
    pq = []
    heapq.heappush(pq, (0, 0, start_word, [start_word]))
    
    visited = {}
    nodes_expanded = 0
    
    while pq:
        f, g, current_word, path = heapq.heappop(pq)
        nodes_expanded += 1
        
        if current_word == goal_word:
            runtime = time.time() - start_time
            return path, len(path), nodes_expanded, runtime
        
        if current_word in visited and visited[current_word] <= g:
            continue
        
        visited[current_word] = g
        
        neighbors = get_k_neighbors(current_word, embeddings, k)
        
        for neighbor in neighbors:
            step_cost = get_path_cost(current_word, neighbor, embeddings)
            new_g = g + step_cost
            h = calculate_heuristic(neighbor, goal_word, embeddings)
            new_f = new_g + h
            
            heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))
    
    return None, 0, nodes_expanded, time.time() - start_time

import time

def main():
    print("Loading embeddings...")
    embeddings = load_embeddings("glove.100d.20000.txt")
    print("Embeddings loaded.\n")
    
    start_word = input("Enter start word: ").strip().lower()
    goal_word = input("Enter goal word: ").strip().lower()
    
    if start_word not in embeddings or goal_word not in embeddings:
        print("Error: One or both words not in vocabulary.")
        return
    
    k = int(input("Enter number of neighbors (k): "))
    
    print("\nChoose algorithm:")
    print("1. BFS")
    print("2. DFS")
    print("3. UCS")
    print("4. Greedy Best-First")
    print("5. A*")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        result = bfs(start_word, goal_word, k, embeddings)
    elif choice == "2":
        depth_limit = int(input("Enter depth limit for DFS: "))
        result = dfs(start_word, goal_word, k, depth_limit, embeddings)
    elif choice == "3":
        result = ucs(start_word, goal_word, k, embeddings)
    elif choice == "4":
        result = greedy_best_first(start_word, goal_word, k, embeddings)
    elif choice == "5":
        result = a_star(start_word, goal_word, k, embeddings)
    else:
        print("Invalid choice.")
        return
    
    path, length, nodes_expanded, runtime = result
    
    print("\n--- Results ---")
    
    if path:
        print("Path found:")
        print(" -> ".join(path))
        print(f"Path length: {length}")
    else:
        print("No path found.")
    
    print(f"Nodes expanded: {nodes_expanded}")
    print(f"Runtime: {runtime:.4f} seconds")
    

if __name__ == "__main__":
    main()
