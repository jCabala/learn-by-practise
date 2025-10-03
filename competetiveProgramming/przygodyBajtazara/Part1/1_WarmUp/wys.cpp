//https://szkopul.edu.pl/problemset/problem/KrDYc6Wu8OK_pwfh9EKVkrr7/site/?key=statement
#include <bits/stdc++.h>
using namespace std;

// Type aliases for readability
using Point = pair<int, int>;
using Edge = pair<Point, Point>;

constexpr int kx[] = {0, 1, 1, 0, -1, -1};
constexpr int ky[] = {1, 1, 0, -1, -1, 0};

// Function to calculate the direction of a given edge
int getDirection(const Edge& edge) {
    for (int i = 0; i < 6; ++i) {
        if (edge.second.first - edge.first.first == kx[i] && 
            edge.second.second - edge.first.second == ky[i]) {
            return i;
        }
    }
    cerr << "Invalid direction";
    exit(1);
}

// Function to move a point in the given direction
Point movePoint(const Point& p, int direction) {
    return {p.first + kx[direction], p.second + ky[direction]};
}

// Function to find the lexicographically smallest rotation of a string
string getSmallestRotation(const string& s) {
    string best = "z";
    int n = s.size();
    string current = s;
    for (int i = 0; i < n; ++i) {
        if (current < best) best = current;
        current = current.substr(1) + current[0];
    }
    return best;
}

// Function to generate the code of a figure given its edges
string encodeFigure(const vector<Edge>& edges) {
    int n = edges.size();
    vector<Edge> cycleEdges = edges;
    cycleEdges.push_back(edges[0]);

    string ret, r1, r2;
    for (int i = 0; i < n; ++i) {
        int diff = (getDirection(cycleEdges[i + 1]) - getDirection(cycleEdges[i]) + 8) % 6;
        ret += char(diff + 'a');
    }

    r1 = getSmallestRotation(ret);
    reverse(ret.begin(), ret.end());
    r2 = getSmallestRotation(ret);

    return min(r1, r2);
}

// Function to create all figure codes derived from a given figure by adding one triangle
vector<string> expandFigure(const vector<Edge>& figure) {
    int n = figure.size();
    vector<Edge> cycleEdges = figure;
    cycleEdges.push_back(figure[0]);

    set<string> uniqueCodes;
    vector<Edge> newFigure;

    for (int i = 0; i < n; ++i) {
        Point nextPoint = movePoint(figure[i].second, (getDirection(figure[i]) + 4) % 6);

        if (nextPoint == figure[(i + n - 1) % n].first && nextPoint != figure[i + 1].second) continue;

        if (nextPoint == figure[(i + n - 1) % n].first && nextPoint == figure[i + 1].second) {
            newFigure.clear();
            for (int j = 0; j < n; ++j) {
                if (j != i && j != (i + 1) % n && j != (i + n - 1) % n) {
                    newFigure.push_back(figure[j]);
                }
            }
            uniqueCodes.insert(encodeFigure(newFigure));
        } else if (nextPoint == figure[i + 1].second && figure[i].first != figure[(i + 2) % n].second) {
            newFigure.clear();
            for (int j = 0; j < n; ++j) {
                if (j != i && j != i + 1) {
                    newFigure.push_back(figure[j]);
                } else if (j == i) {
                    newFigure.push_back({figure[i].first, figure[i + 1].second});
                }
            }
            uniqueCodes.insert(encodeFigure(newFigure));
        } else if (nextPoint != figure[i + 1].second) {
            int currentDirection = getDirection(figure[i]);
            newFigure.clear();
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    newFigure.push_back(figure[j]);
                } else {
                    Point newPoint = movePoint(figure[i].first, (currentDirection + 5) % 6);
                    newFigure.push_back({figure[i].first, newPoint});
                    newFigure.push_back({newPoint, figure[i].second});
                }
            }
            uniqueCodes.insert(encodeFigure(newFigure));
        }
    }

    return {uniqueCodes.begin(), uniqueCodes.end()};
}

// Function to reconstruct a figure from its code
vector<Edge> decodeFigure(const string& code) {
    int n = code.size() - 1;
    vector<Edge> edges;

    int direction = 0;
    Point current = {0, 1};
    edges.push_back({{0, 0}, current});

    for (int i = 0; i < n; ++i) {
        direction = (direction + int(code[i] - 'a') - 2 + 6) % 6;
        Point nextPoint = movePoint(current, direction);
        edges.push_back({current, nextPoint});
        current = nextPoint;
    }

    return edges;
}

set<string> results[11];

// Generate all figures for the task
void generateAllFigures() {
    vector<Point> points = {{0, 0}, {1, 1}, {1, 0}, {0, 0}};
    vector<Edge> initialFigure;

    for (int i = 0; i < 3; ++i) {
        initialFigure.push_back({points[i], points[i + 1]});
    }

    results[1].insert("eee");
    for (int i = 2; i <= 10; ++i) {
        for (const auto& code : results[i - 1]) {
            vector<string> newFigures = expandFigure(decodeFigure(code));
            results[i].insert(newFigures.begin(), newFigures.end());
        }
    }
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0);
    int t, n;
    char type;
    char description[50];

    generateAllFigures();

    cin >> t;
    for (int i = 0; i < t; ++i) {
        cin >> type;
        if (type == 'K') {
            cin >> description;
            vector<string> expandedFigures = expandFigure(decodeFigure(description));
            cout << expandedFigures.size() << "\n";
            for (size_t j = 0; j < expandedFigures.size(); ++j) {
                if (j > 0) cout << " ";
                cout << expandedFigures[j];
            }
            cout << "\n";
        } else {
            cin >> n;
            cout << results[n].size() << "\n";
            for (auto it = results[n].begin(); it != results[n].end(); ++it) {
                if (it != results[n].begin()) cout << " ";
                cout << *it;
            }
            cout << "\n";
        }
    }
    return 0;
}

