# LangGraph SDK Documentation

This documentation explains the concepts and usage of the `@langchain/langgraph-sdk` package. It provides an overview of all classes, methods, and properties, along with examples and their integration with a Next.js frontend using the App Router. A real-world example is included at the end.

---

## Installation

To use the LangGraph SDK, install it via npm or yarn:

```bash
npm install @langchain/langgraph-sdk
```

---

## Classes

### **Client**

The `Client` class acts as the main entry point for the LangGraph SDK, bundling all sub-clients (like Assistants, Threads, Store, Crons, and Runs) into a single instance. You can use this class to manage operations across the SDK's functionalities.

#### **Constructor**

```typescript
import { Client } from "@langchain/langgraph-sdk";

const client = new Client(config);
```
- **`config`** *(optional)*: `ClientConfig` object for configuration.

#### **Properties**

- **`assistants`**: Instance of `AssistantsClient`.
- **`threads`**: Instance of `ThreadsClient`.
- **`store`**: Instance of `StoreClient`.
- **`runs`**: Instance of `RunsClient`.
- **`crons`**: Instance of `CronsClient`.

#### **Methods**

##### `deleteItem(namespace, key)`
Deletes an item from the key-value store.

**Parameters**:
- `namespace`: An array of strings representing the namespace path.
- `key`: The unique identifier for the item to be deleted.

**Returns**: `Promise<void>`

**Example**:

```typescript
await client.store.deleteItem(["namespace1", "namespace2"], "itemKey");
console.log("Item deleted successfully.");
```

##### `getItem(namespace, key)`
Retrieves a single item from the key-value store.

**Parameters**:
- `namespace`: An array of strings representing the namespace path.
- `key`: The unique identifier for the item to be retrieved.

**Returns**: `Promise<null | Item>`

**Example**:

```typescript
const item = await client.store.getItem(["namespace1"], "itemKey");
console.log("Retrieved item:", item);
```

##### `listNamespaces(options?)`
Lists namespaces with optional filters and pagination.

**Parameters**:
- `options?`:
  - `limit?`: *(optional)* Maximum number of namespaces to return (default: 100).
  - `maxDepth?`: *(optional)* Maximum depth of namespaces to include.
  - `offset?`: *(optional)* Number of namespaces to skip before returning results (default: 0).
  - `prefix?`: *(optional)* List of strings to filter namespaces by prefix.
  - `suffix?`: *(optional)* List of strings to filter namespaces by suffix.

**Returns**: `Promise<ListNamespaceResponse>`

**Example**:

```typescript
const namespaces = await client.store.listNamespaces({ limit: 10 });
console.log("Namespaces:", namespaces);
```

##### `putItem(namespace, key, value)`
Stores or updates an item in the key-value store.

**Parameters**:
- `namespace`: An array of strings representing the namespace path.
- `key`: The unique identifier for the item.
- `value`: A record object containing the item's data.

**Returns**: `Promise<void>`

**Example**:

```typescript
await client.store.putItem(["namespace1"], "itemKey", { data: "example" });
console.log("Item stored successfully.");
```

##### `searchItems(namespacePrefix, options?)`
Searches for items within a namespace prefix.

**Parameters**:
- `namespacePrefix`: An array of strings representing the namespace prefix.
- `options?`:
  - `filter?`: *(optional)* Dictionary of key-value pairs to filter results.
  - `limit?`: *(optional)* Maximum number of items to return (default: 10).
  - `offset?`: *(optional)* Number of items to skip before returning results (default: 0).
  - `query?`: *(optional)* Search query string.

**Returns**: `Promise<SearchItemsResponse>`

**Example**:

```typescript
const items = await client.store.searchItems(["namespace1"], { limit: 5 });
console.log("Searched items:", items);
```

---

### **AssistantsClient**

The `AssistantsClient` allows managing assistants created in your Python application. Instead of creating new assistants, you can retrieve and interact with existing ones using their `assistantId`.

#### Example: Retrieve an Existing Assistant

```typescript
const assistant = await client.assistants.get("existing-assistant-id");
console.log("Retrieved Assistant:", assistant);
```

---

### **ThreadsClient**

Operations related to threads can be accessed via the `threads` property of the `Client` class. Threads are used for managing states or operations within your application.

**Example**:

```typescript
const thread = await client.threads.create({
  metadata: { purpose: "example-thread" },
});
console.log("Created Thread:", thread);
```

---

### **CronsClient**

Scheduled task operations are accessible via the `crons` property of the `Client` class.

**Example**:

```typescript
const cronJob = await client.crons.create("assistant-id-123", {
  interval: "0 * * * *",
  metadata: { task: "hourly-task" },
});
console.log("Created Cron Job:", cronJob);
```

---

## Example: Integrating with Next.js (App Router)

### Setting Up
1. Install dependencies:

```bash
npm install @langchain/langgraph-sdk
```

2. Create an API route in your Next.js App Router (`/app/api/threads/route.ts`):

```typescript
import { Client } from "@langchain/langgraph-sdk";

const client = new Client();

export async function POST(request: Request) {
  const body = await request.json(); // Parse JSON body
  const { threadId } = body; // Extract threadId from request

  try {
    const thread = await client.threads.get(threadId); // Retrieve thread using its ID
    return new Response(JSON.stringify(thread), { status: 200 }); // Return the retrieved thread
  } catch (error) {
    console.error("Error retrieving thread:", error); // Log errors if any
    return new Response("Failed to retrieve thread", { status: 500 }); // Return error response
  }
}
```

### Frontend Example
Interact with existing threads:

```typescript
'use client';

import { useState } from "react";

export default function FetchThread() {
  const [threadId, setThreadId] = useState(""); // State to hold thread ID
  const [thread, setThread] = useState(null); // State to hold thread data

  const handleFetch = async () => {
    const res = await fetch("/api/threads", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ threadId }), // Send threadId in the request body
    });

    const data = await res.json(); // Parse the response
    setThread(data); // Update thread state with retrieved data
  };

  return (
    <div>
      <input
        type="text"
        placeholder="Thread ID"
        value={threadId}
        onChange={(e) => setThreadId(e.target.value)} // Update threadId state on input change
      />
      <button onClick={handleFetch}>Fetch Thread</button> {/* Button to fetch thread */}
      {thread && <pre>{JSON.stringify(thread, null, 2)}</pre>} {/* Display thread details */}
    </div>
  );
}
```

---

With this documentation, you can easily understand and utilize the LangGraph SDK to interact with existing assistants and threads in your Python-based applications while building robust integrations in Next.js.
