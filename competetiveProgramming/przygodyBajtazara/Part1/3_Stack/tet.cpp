// https://szkopul.edu.pl/problemset/problem/noPY-IL0vsAi2TiXF-v2f5Br/site/?key=statement
#include <bits/stdc++.h>
using namespace std;

int n;
vector<int> vec;

const int MAXN = 50001;

bool inStack[MAXN];

// Key point: "Ponadto dane testowe są tak dobrane, że istnieje rozwiązanie zawierające nie więcej niż  ruchów.""

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> n;
    for (int i = 0; i < 2 * n; i++) {
        int x; cin >> x;
        vec.push_back(x);
    }

    stack<int> st, temp;
    vector<int> moves;

    for (int i = 0; i < 2 * n; i++) {
        int cur = vec[i];
        if (!inStack[cur]) {
            st.push(cur);
            inStack[cur] = true;
        } else {
            while (st.top() != cur) {
                moves.push_back(st.size());
                temp.push(st.top());
                st.pop();
            }

            st.pop();
            while (!temp.empty()) {
                st.push(temp.top());
                temp.pop();
            }
        }
    } 

    cout << moves.size() << "\n";
    for (int x : moves) cout << x << "\n";
    cout << "\n";
}