//https://szkopul.edu.pl/problemset/problem/TcT3eHfPGmUV2NmvzqyvVn9e/site/?key=statement
#include <bits/stdc++.h>

using namespace std;

bool DEBUG = false;

const int MAXN = 50005, MAX_HEIGHT = 100005;
int series[2][MAXN];
bool visited[2][MAXN];
vector<pair<int,int>> indexes[MAX_HEIGHT];

bool has_twin(int x, int y) {
    int current_height = series[x][y];
    auto& vec = indexes[current_height];
    
    return vec.size() == 2;
}

pair<int, int> get_twin(int x, int y) {
    int current_height = series[x][y];
    auto& vec = indexes[current_height];
    
    if (vec[0] == make_pair(x, y)) {
        return vec[1];
    } else {
        return vec[0];
    }
}

struct TraverseResult {
    int changes_if_changed;
    int length;
    bool is_good; // If true it means that all pairs are on different sides
};
TraverseResult traverse(int x, int y, bool change) {
    if (visited[x][y]) {
        return {0, 0, true};
    }
    visited[x][y] = true;

    // Change sides
    auto cur = make_pair(1 - x, y);
    visited[cur.first][cur.second] = true;

    // If there is no twin, we stop here
    if (!has_twin(cur.first, cur.second)) {
        return {change ? 1 : 0, 1, true};
    }

    // There is a twin, we continue
    auto twin = get_twin(cur.first, cur.second);

    // If twin is on the same side, `change` switches otherwisе it remains the same
    bool new_change = (twin.first == cur.first) ? !change : change;
    auto res = traverse(twin.first, twin.second, new_change);
    return {res.changes_if_changed + (change ? 1 : 0), res.length + 1, res.is_good && (twin.first != cur.first)};
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n;
    cin >> n;

    for (int i = 0; i < n; i++) {
        cin >> series[0][i];
        indexes[series[0][i]].push_back({0, i});
    }

    for (int i = 0; i < n; i++) {
        cin >> series[1][i];
        indexes[series[1][i]].push_back({1, i});
    }

    int ans = 0;
    for (int i = 0; i < n; i++) {
        auto r1 = traverse(0, i, true);
        TraverseResult r2 = {0, 0, true};

        bool first_pair_good = true;
        if (has_twin(0, i)) {
            auto twin = get_twin(0, i);
            r2 = traverse(twin.first, twin.second, twin.first == 1);
            first_pair_good = (twin.first != 0);
        }

        if (DEBUG) cout << i << "|  " << r1.changes_if_changed << " " << r1.length << " " << r1.is_good << " | " << r2.changes_if_changed << " " << r2.length << " " << r2.is_good << " | " << first_pair_good << "\n";

        if (r1.is_good && r2.is_good && first_pair_good) continue;

        int val_if_changed = r1.changes_if_changed + r2.changes_if_changed;
        int min_val = min(val_if_changed, r1.length + r2.length - val_if_changed);
        if (DEBUG) cout << "Min val for " << i << ": " << min_val << "\n";
        ans += min_val;
    }

    cout << ans << "\n";
    return 0;
}