# CLAUDE.md — 라이브 아티팩트 개발 가이드

## callMcpTool 반환 포맷

라이브 아티팩트(HTML) 내에서 `window.callMcpTool(toolName, params)`를 호출하면
**MCP 프로토콜 형식**으로 감싸진 응답이 반환됩니다. 직접 사용하면 오류가 발생하므로
**반드시** 아래 `parseResp`로 파싱해야 합니다.

```
반환 포맷:
{
  content: [
    { type: 'text', text: '<JSON 문자열 또는 평문>' }
  ]
}
```

## 표준 유틸 — 모든 라이브 아티팩트에 포함

```javascript
// callMcpTool 실제 반환 포맷 파싱
function parseResp(raw) {
  if (!raw) return null;
  if (raw.content && Array.isArray(raw.content) && raw.content[0]?.type === 'text') {
    try { return JSON.parse(raw.content[0].text); } catch { return raw.content[0].text; }
  }
  return raw;
}

// MCP 호출 래퍼 — callMcpTool 없는 환경에서는 null 반환
async function callMcp(toolName, params = {}) {
  try {
    if (typeof window.callMcpTool === 'function') {
      const raw = await window.callMcpTool(toolName, params);
      return parseResp(raw);
    }
  } catch (e) {
    console.warn(`MCP "${toolName}" 호출 실패:`, e);
  }
  return null; // 호출부에서 fallback 처리
}
```

## 사용 패턴

```javascript
async function loadData() {
  const live = await callMcp('tool_name', { param: 'value' });
  if (live) {
    // 실제 MCP 데이터 사용
    render(live);
  } else {
    // callMcpTool 미지원 환경 또는 호출 실패 → 모의 데이터 fallback
    render(MOCK_DATA);
  }
}
```

## 주의사항

- `callMcpTool`은 Claude Code 웹 환경에서만 주입됩니다. 로컬 파일로 열면 항상 `null`입니다.
- MCP 데이터와 모의 데이터 스키마를 동일하게 맞춰 fallback이 매끄럽게 동작하도록 합니다.
- `renderXxx` 함수가 MCP 호출을 포함하면 반드시 `async`로 선언하고 `init`에서 `await`합니다.
