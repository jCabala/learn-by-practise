// https://szkopul.edu.pl/problemset/problem/A7ZI0Wwn6tTiCJoYblblTAqz/site/?key=statement
#include <bits/stdc++.h>

using namespace std;

// Big integer class for handling large powers of 2
class BigInt {
private:
    vector<int> digits; // Store digits in reverse order (least significant first)
    static const int BASE = 1000000000; // 10^9
    
public:
    BigInt(int n = 0) {
        if (n == 0) {
            digits.push_back(0);
        } else {
            while (n > 0) {
                digits.push_back(n % BASE);
                n /= BASE;
            }
        }
    }
    
    BigInt operator*(int n) const {
        BigInt result;
        result.digits.clear();
        
        long long carry = 0;
        for (int i = 0; i < digits.size(); i++) {
            long long prod = (long long)digits[i] * n + carry;
            result.digits.push_back(prod % BASE);
            carry = prod / BASE;
        }
        
        while (carry > 0) {
            result.digits.push_back(carry % BASE);
            carry /= BASE;
        }
        
        return result;
    }
    
    void print() const {
        cout << digits.back();
        for (int i = digits.size() - 2; i >= 0; i--) {
            // Print with leading zeros
            int d = digits[i];
            for (int div = BASE / 10; div > 0; div /= 10) {
                cout << (d / div) % 10;
            }
        }
        cout << endl;
    }
};

// Node in the dependency graph
struct Node {
    char var; // variable name, '0' for digit 0, '1' for digit 1
    int pos;  // position in the variable expansion (1-indexed)
    
    Node(char v = 0, int p = 0) : var(v), pos(p) {}
    
    bool operator<(const Node& other) const {
        if (var != other.var) return var < other.var;
        return pos < other.pos;
    }
    
    bool operator==(const Node& other) const {
        return var == other.var && pos == other.pos;
    }
};

// DFS to explore connected component
void dfs(const Node& node, 
         const map<Node, vector<Node>>& graph,
         map<Node, bool>& visited,
         bool& hasZero,
         bool& hasOne) {
    visited[node] = true;
    
    if (node.var == '0') hasZero = true;
    if (node.var == '1') hasOne = true;
    
    if (graph.find(node) != graph.end()) {
        for (const Node& neighbor : graph.at(node)) {
            if (!visited[neighbor]) {
                dfs(neighbor, graph, visited, hasZero, hasOne);
            }
        }
    }
}

// Count connected components in the graph
// Returns -1 if there's a contradiction (no solution)
// Returns the number of free components otherwise
int countComponents(const map<Node, vector<Node>>& graph,
                   const vector<Node>& allNodes) {
    map<Node, bool> visited;
    int freeComponents = 0;
    
    for (const Node& node : allNodes) {
        if (!visited[node]) {
            bool hasZero = false;
            bool hasOne = false;
            
            dfs(node, graph, visited, hasZero, hasOne);
            
            // Check if component has both 0 and 1
            if (hasZero && hasOne) {
                return -1; // Contradiction - no solution
            }
            
            // Count only components without 0 or 1
            if (!hasZero && !hasOne) {
                freeComponents++;
            }
        }
    }
    
    return freeComponents;
}

int solveEquation() {
    int k; // number of variables
    cin >> k;
    
    vector<int> varLength(26, 0);
    for (int i = 0; i < k; i++) {
        cin >> varLength[i];
    }
    
    // Read left side
    int l;
    cin >> l;
    string left;
    vector<Node> leftExpansion;
    for (int i = 0; i < l; i++) {
        char c;
        cin >> c;
        if (c == '0' || c == '1') {
            leftExpansion.push_back(Node(c, 0));
        } else {
            // Variable
            int len = varLength[c - 'a'];
            for (int j = 1; j <= len; j++) {
                leftExpansion.push_back(Node(c, j));
            }
        }
    }
    
    // Read right side
    int r;
    cin >> r;
    vector<Node> rightExpansion;
    for (int i = 0; i < r; i++) {
        char c;
        cin >> c;
        if (c == '0' || c == '1') {
            rightExpansion.push_back(Node(c, 0));
        } else {
            // Variable
            int len = varLength[c - 'a'];
            for (int j = 1; j <= len; j++) {
                rightExpansion.push_back(Node(c, j));
            }
        }
    }
    
    // Check if expansions have the same length
    if (leftExpansion.size() != rightExpansion.size()) {
        return -1; // No solution
    }
    
    // Build dependency graph using adjacency list
    map<Node, vector<Node>> graph;
    vector<Node> allNodes;
    
    for (int i = 0; i < leftExpansion.size(); i++) {
        Node left = leftExpansion[i];
        Node right = rightExpansion[i];
        
        // Add edges in both directions (undirected graph)
        graph[left].push_back(right);
        graph[right].push_back(left);
        
        allNodes.push_back(left);
        allNodes.push_back(right);
    }
    
    // Count components using DFS
    int freeComponents = countComponents(graph, allNodes);
    
    return freeComponents;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int x; // number of equations
    cin >> x;
    
    for (int i = 0; i < x; i++) {
        int freeComponents = solveEquation();
        
        if (freeComponents == -1) {
            // No solution (contradiction)
            cout << 0 << endl;
        } else {
            // Calculate 2^freeComponents using BigInt
            BigInt result(1);
            for (int j = 0; j < freeComponents; j++) {
                result = result * 2;
            }
            result.print();
        }
    }
    
    return 0;
}
