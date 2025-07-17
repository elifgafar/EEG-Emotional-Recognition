## Performance Testing

The API performance was evaluated using ApacheBench (ab) with the following configuration:

- Total Requests: 100  
- Concurrency Level: 10  
- Request Type: POST `/predict` with JSON payload

### Test Results Summary

| Metric                          | Value             | Description                                            |
|---------------------------------|-------------------|--------------------------------------------------------|
| Total Completed Requests        | 100               | All requests completed successfully                    |
| Failed Requests                 | 0                 | No failed requests                                     |
| Requests per Second             | 78.47 [#/sec]     | Number of requests handled per second (throughput)     |
| Average Time per Request        | 127.44 ms         | Mean time to process a single request                  |
| Average Time per Request (concurrent) | 12.74 ms    | Mean time per request across all concurrent requests   |
| Transfer Rate                   | 13.49 Kbytes/sec received <br> 5641.80 Kbytes/sec sent | Network throughput|
| Longest Request Duration        | 158 ms            | Maximum time taken by a single request                 |

### Connection and Processing Times (ms)

| Phase        | Minimum | Mean (±SD) | Median | Maximum |
|--------------|---------|------------|--------|---------|
| Connect      | 0       | 0 ± 0.0    | 0      | 0       |
| Processing   | 14      | 123 ± 22.6 | 124    | 158     |
| Waiting      | 14      | 95 ± 29.5  | 101    | 140     |
| Total        | 14      | 123 ± 22.6 | 124    | 158     |

### Request Completion Percentiles (ms)

| Percentile | Time (ms) |
|------------|-----------|
| 50%        | 124       |
| 66%        | 137       |
| 75%        | 138       |
| 80%        | 138       |
| 90%        | 138       |
| 95%        | 147       |
| 98%        | 147       |
| 99%        | 158       |
| 100%       | 158       |

---

### Evaluation

The API demonstrates solid performance under moderate concurrency, maintaining low latency and zero failed requests. With an average request processing time around 127 ms and throughput of ~78 requests per second, it is suitable for real-time or near-real-time applications.


### API Testing with Swagger UI

FastAPI automatically provides an interactive API documentation powered by **Swagger UI**.  
This makes it extremely easy to explore and test the `/predict` endpoint right from your browser.


#### 🔎 Interactive API Testing with Swagger UI

Once the FastAPI server is running, you can access the interactive API documentation provided by Swagger UI at:

http://127.0.0.1:8000/docs

Features:
📎 Browse all available API endpoints.
📎 Submit requests directly from the browser.
📎 View and test responses in real-time.

To test the /predict endpoint:

1. pen http://127.0.0.1:8000/docs in your browser.

2. Click on the POST /predict section to expand it.

3. Paste your input (e.g., from sample_input.json) into the request body field.

4. Click “Execute”.

5. View the prediction response directly in the browser

💡 This is an easy and user-friendly way to test and demonstrate your model predictions without needing external tools like Postman or cURL. 💡