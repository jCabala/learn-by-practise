// https://szkopul.edu.pl/problemset/problem/fYXVXOreVxlXTRoHZJXyXF2l/site/?key=statement
#include <bits/stdc++.h>
using namespace std;

int binSer(vector<int> &pipe, int disk, int curPos) {
    int l = 0, r = curPos-1, ans = -1;
    while (l <= r) {
        int mid = (l+r)/2;
        if (pipe[mid] >= disk) {
            ans = mid;
            l = mid + 1;
        } else {
            r = mid-1;
        }
    }
    return ans;
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(NULL);
    vector<int> pipe;
    int n, m;
    cin >> n >> m;
    pipe.resize(n);
    for (int i = 0; i < n; i++) {
        cin >> pipe[i];
        if (i > 0) {
            pipe[i] = min(pipe[i], pipe[i-1]);
        }
    }

    int curPos = n, disk;
    bool stuck = false;
    for (int i = 0; i < m; i++) {
        if (curPos == 0) {
            stuck = true;
        }
        cin >> disk;
        curPos = binSer(pipe, disk, curPos);
    }

    stuck ? cout << "0\n" : cout << curPos + 1 << "\n";
}
