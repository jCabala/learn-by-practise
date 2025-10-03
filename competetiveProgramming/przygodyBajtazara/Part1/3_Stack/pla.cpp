// https://szkopul.edu.pl/problemset/problem/au-E9FH96-3U9rCKhcNsD5n9/site/?key=statement
#include <bits/stdc++.h>
using namespace std;

const bool DEBUG = false;
const int MAXN =  250001;
int n, w, heights[MAXN], ans = 0;
stack<int> s;

void printDebug() {
    if (!DEBUG) return;

    stack<int> s2 = s;
    cout << "\nans: " << ans << "\n";
    cout << "s: ";
    while(!s2.empty()) {
        cout << s2.top() << " ";
        s2.pop();
    }
    cout << "\n";
}

int main() {
    ios::sync_with_stdio(0); cin.tie(0);
    cin >> n;
    for (int i = 0; i < n; i++) {
        cin >> w >> heights[i];
    }

    for(int i = 0; i < n; i++) {
        int h = heights[i];
        if (s.empty() || h > s.top()) {
            ans++;
            s.push(h);
        } else if (h < s.top()) {
            while(!s.empty() && h < s.top()) s.pop();
            i--;
        }
        printDebug();
    }

    cout << ans << "\n";
}