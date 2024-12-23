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

### **AssistantsClient**

This client handles operations related to creating, managing, and retrieving assistants.

#### **Constructor**

```typescript
const assistantsClient = new AssistantsClient(config);
```
- **`config`** *(optional)*: `ClientConfig` object for configuration.

#### **Methods**

##### `create(payload)`
Creates a new assistant.

**Parameters**:
- `payload`:
  - `assistantId?`: *(optional)* ID of the assistant.
  - `config?`: *(optional)* Configuration for the assistant.
  - `graphId`: ID of the graph.
  - `ifExists?`: *(optional)* Behavior if assistant already exists.
  - `metadata?`: *(optional)* Metadata.
  - `name?`: *(optional)* Name of the assistant.

**Returns**: `Promise<Assistant>`

**Example**:

```typescript
const assistant = await assistantsClient.create({
  graphId: "graph-id-123",
  name: "My Assistant",
});
console.log("Created Assistant:", assistant); // Log the created assistant details
```

##### `delete(assistantId)`
Deletes an assistant by ID.

**Parameters**:
- `assistantId`: ID of the assistant.

**Returns**: `Promise<void>`

**Example**:

```typescript
await assistantsClient.delete("assistant-id-123");
console.log("Assistant deleted."); // Log confirmation after deletion
```

##### `get(assistantId)`
Retrieves an assistant by ID.

**Parameters**:
- `assistantId`: ID of the assistant.

**Returns**: `Promise<Assistant>`

**Example**:

```typescript
const assistant = await assistantsClient.get("assistant-id-123");
console.log("Assistant Details:", assistant); // Display assistant details
```

---

### **Client**

The `Client` class acts as the main entry point for the LangGraph SDK, allowing access to sub-clients like Assistants, Threads, Store, Crons, and Runs.

#### **Constructor**

```typescript
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

### **ThreadsClient**

Handles operations related to threads.

#### **Methods**

##### `create(payload)`
Creates a new thread.

**Parameters**:
- `payload`: Configuration for creating a thread.

**Returns**: `Promise<Thread>`

**Example**:

```typescript
const thread = await client.threads.create({
  metadata: { purpose: "example-thread" },
});
console.log("Created Thread:", thread); // Log created thread details
```

##### `get(threadId)`
Retrieves a thread by ID.

**Parameters**:
- `threadId`: ID of the thread.

**Returns**: `Promise<Thread>`

**Example**:

```typescript
const thread = await client.threads.get("thread-id-123");
console.log("Thread Details:", thread); // Log retrieved thread details
```

---

## Example: Integrating with Next.js (App Router)

### Setting Up
1. Install dependencies:

```bash
npm install @langchain/langgraph-sdk
```

2. Create an API route in your Next.js App Router (`/app/api/assistants/route.ts`):

```typescript
import { AssistantsClient } from "@langchain/langgraph-sdk";

const assistantsClient = new AssistantsClient();

export async function POST(request: Request) {
  const body = await request.json(); // Parse JSON body
  const { name, graphId } = body; // Extract name and graphId from request

  try {
    const assistant = await assistantsClient.create({
      name, // Pass name to the create method
      graphId, // Pass graphId to the create method
    });
    return new Response(JSON.stringify(assistant), { status: 201 }); // Return the created assistant
  } catch (error) {
    console.error("Error creating assistant:", error); // Log errors if any
    return new Response("Failed to create assistant", { status: 500 }); // Return error response
  }
}
```

### Frontend Example
Create a form to create an assistant:

```typescript
'use client';

import { useState } from "react";

export default function CreateAssistant() {
  const [name, setName] = useState(""); // State to hold assistant name
  const [graphId, setGraphId] = useState(""); // State to hold graph ID

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault(); // Prevent default form submission

    const res = await fetch("/api/assistants", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ name, graphId }), // Send name and graphId in the request body
    });

    const data = await res.json(); // Parse the response
    console.log("Created Assistant:", data); // Log the response
  };

  return (
    <form onSubmit={handleSubmit}> {/* Form to capture assistant details */}
      <input
        type="text"
        placeholder="Assistant Name"
        value={name}
        onChange={(e) => setName(e.target.value)} // Update name state on input change
      />
      <input
        type="text"
        placeholder="Graph ID"
        value={graphId}
        onChange={(e) => setGraphId(e.target.value)} // Update graphId state on input change
      />
      <button type="submit">Create Assistant</button> {/* Submit button */}
    </form>
  );
}
```

---

## Real-World Example: Automated Task Management

Use LangGraph SDK to manage tasks, create threads, and utilize assistants for automating workflows.

### Backend

1. **Create a task automation graph**:

```typescript
import { Client } from "@langchain/langgraph-sdk";

const client = new Client();

async function createTaskAutomation() {
  const assistant = await client.assistants.create({
    graphId: "task-graph-id", // Specify the graph ID for automation
    name: "Task Automation Assistant", // Specify the assistant name
  });

  console.log("Assistant Created:", assistant); // Log the created assistant
}

createTaskAutomation();
```

2. **Trigger runs for tasks**:

```typescript
async function runTask(assistantId, threadId) {
  const run = await client.runs.create(threadId, assistantId, {
    payload: {
      task: "Automate daily report generation", // Specify the task details
    },
  });

  console.log("Run Created:", run); // Log the created run
}
```

### Frontend

Display ongoing tasks and statuses using the ThreadsClient:

```typescript
import { useEffect, useState } from "react";

export default function TaskManager() {
  const [tasks, setTasks] = useState([]); // State to hold tasks

  useEffect(() => {
    async function fetchTasks() {
      const res = await fetch("/api/threads"); // Fetch thread data from API
      const data = await res.json(); // Parse the response
      setTasks(data); // Update tasks state with fetched data
    }

    fetchTasks(); // Call the function on component mount
  }, []);

  return (
    <div>
      <h1>Task Manager</h1>
      <ul>
        {tasks.map((task) => (
          <li key={task.id}>{task.metadata.purpose}</li> // Render task purposes
        ))}
      </ul>
    </div>
  );
}
```

---

With this documentation, you can easily understand and utilize the LangGraph SDK to build robust applications integrated with Next.js.
