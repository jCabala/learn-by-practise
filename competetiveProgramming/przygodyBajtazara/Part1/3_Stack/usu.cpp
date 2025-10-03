// https://szkopul.edu.pl/problemset/problem/WXorPFSPZwmWfJWy1GSuJ9QD/site/?key=statement
#include <bits/stdc++.h>
using namespace std;

const int MAXN = 1000005;

int n, k;
int arr[MAXN], prefSum[MAXN];
vector<vector<int>> moves;

void printMove(const vector<int> &move) {
    for (int x : move) {
        cout << x << " ";
    }
    cout << "\n";
}

void makeMove(stack<int> &s) {
    // Make move using k + 1 top blocks
    auto move = vector<int>();
    for (int i = 0; i < k + 1; i++) {
        move.push_back(s.top());
        s.pop();
    }

    reverse(move.begin(), move.end());
    moves.push_back(move);
}

int main() {
    ios::sync_with_stdio(0); cin.tie(0);
    cin >> n >> k;
    for (int i = 1; i <= n; i++) {
        char c;
        cin >> c;
        arr[i] = (c == 'b' ? 1 : -k); // b = 1, c = -k
    }

    stack<int> s;

    for (int i = 1; i <= n; i++) {
        s.push(i);
        prefSum[s.size()] = prefSum[s.size() - 1] + arr[i];

        if (s.size() >= k + 1 && prefSum[s.size()] - prefSum[s.size() - k - 1] == 0) {
            makeMove(s);
        }
    }

    reverse(moves.begin(), moves.end());
    for (int i = 0; i < (int)moves.size(); i++) {
        printMove(moves[i]);
    }
}