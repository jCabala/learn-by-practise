// https://szkopul.edu.pl/problemset/problem/Pi-gX_dH2TzvdxTc74KjdHLo/site/?key=statement
#include <bits/stdc++.h>

using namespace std;

struct Point {
    int x, y;
    
    Point(int x = 0, int y = 0) : x(x), y(y) {}

    bool operator<(const Point& other) const {
        return tie(x, y) < tie(other.x, other.y);
    }

    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

class Rectangle {
    public:
    Point topRight, bottomLeft;

    Rectangle(int blx_, int bly_, int trx_, int try_) : bottomLeft(blx_, bly_), topRight(trx_, try_) {}

    bool intersects(const Rectangle& other) const {
        // If they have only one common point, we DO NOT consider it as intersecting
        // If they have a common edge, we DO consider it as intersecting

        // First check if rectangles are completely separate
        if (topRight.x < other.bottomLeft.x || other.topRight.x < bottomLeft.x ||
            topRight.y < other.bottomLeft.y || other.topRight.y < bottomLeft.y) {
            return false; // No intersection
        }
        
        // Check if they only touch at a single corner point
        // This happens when exactly one x-coordinate and one y-coordinate align at endpoints
        bool touchAtX = (topRight.x == other.bottomLeft.x) || (bottomLeft.x == other.topRight.x);
        bool touchAtY = (topRight.y == other.bottomLeft.y) || (bottomLeft.y == other.topRight.y);
        
        if (touchAtX && touchAtY) {
            return false; // Only one common corner point
        }
        
        return true; // Rectangles intersect (share edge or area)
    }
};


vector<vector<int>> create_edges(const vector<Rectangle>& rectangles) {
    int n = rectangles.size();
    vector<vector<int>> edges(n);

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (rectangles[i].intersects(rectangles[j])) {
                edges[i].push_back(j);
                edges[j].push_back(i);
            }
        }
    }

    return edges;
}

int get_connected_components(const vector<vector<int>>& edges) {
    int n = edges.size();
    vector<bool> visited(n, false);
    int component_count = 0;

    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            component_count++;
            queue<int> q;
            q.push(i);
            visited[i] = true;

            while (!q.empty()) {
                int node = q.front();
                q.pop();

                for (int neighbor : edges[node]) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
        }
    }

    return component_count;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    cin >> n;
    vector<Rectangle> rectangles;
    for (int i = 0; i < n; i++) {
        int blx, bly, trx, try_;
        cin >> blx >> bly >> trx >> try_;
        rectangles.emplace_back(blx, bly, trx, try_);
    }

    vector<vector<int>> edges = create_edges(rectangles);
    
    int num_of_connected_components = get_connected_components(edges);

    cout << num_of_connected_components << endl;

    return 0;
}