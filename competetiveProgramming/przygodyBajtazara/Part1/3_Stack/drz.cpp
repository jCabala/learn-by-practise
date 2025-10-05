// https://szkopul.edu.pl/problemset/problem/7-HJKUXJXg7Fcq0WLy5i1hhT/site/?key=statement
#include <bits/stdc++.h>
using namespace std;

const int MAXN = 2501;
int n, levels[MAXN];

// Variables useful for traversal
int lvlPtr = 0;
int nextNodeNumber = 1;

vector<int> genealogyOrder;
vector<char> parenthasesOrder;

void buildOrders(int parentNumber, int curentDepth) {
    if (lvlPtr >= n) return;

    genealogyOrder.push_back(parentNumber);
    parenthasesOrder.push_back('(');
    int myNumber = nextNodeNumber++;

    if (curentDepth == levels[lvlPtr]) {
        lvlPtr++;
    } else {
        buildOrders(myNumber, curentDepth + 1);
        buildOrders(myNumber, curentDepth + 1);
    }

    parenthasesOrder.push_back(')');
}


int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> n;
    for (int i = 0; i < n; i++) cin >> levels[i];

    stack<int> st;
    for (int i = 0; i < n; i++) {
        int cur = levels[i];
        while (!st.empty() && cur > 0 && st.top() == cur) {
            st.pop();
            cur--;
        }
        
        if (cur > 0) st.push(cur);
    }

    if (!st.empty()) {
        cout << "NIE\n";
    } else {
        buildOrders(0, 0);
        for (int num : genealogyOrder) cout << num << " ";
        cout << "\n";

        for (char c : parenthasesOrder) cout << c;
        cout << "\n";
    }
}