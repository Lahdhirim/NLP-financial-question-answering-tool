import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class PredictionService {
  
  private apiUrl = 'http://127.0.0.1:5000';

  constructor(private http: HttpClient) {}

  getPrediction(data: { context: string; question: string }): Observable<{ answer: string }> {
    return this.http.post<{ answer: string }>(`${this.apiUrl}/predict`, data);
  }

  shutdownServer(): Observable<any> {
    return this.http.post(`${this.apiUrl}/shutdown`, {});
  }
}
