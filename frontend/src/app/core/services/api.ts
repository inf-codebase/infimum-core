import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, catchError, throwError } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private http = inject(HttpClient);
  // In a real app, use environment.apiUrl
  private apiUrl = 'api';

  get<T>(path: string, options: any = {}): Observable<T> {
    return this.http.get<T>(`${this.apiUrl}/${path}`, options).pipe(
      catchError(this.handleError)
    ) as Observable<T>;
  }

  post<T>(path: string, body: any, options: any = {}): Observable<T> {
    return this.http.post<T>(`${this.apiUrl}/${path}`, body, options).pipe(
      catchError(this.handleError)
    ) as Observable<T>;
  }

  put<T>(path: string, body: any, options: any = {}): Observable<T> {
    return this.http.put<T>(`${this.apiUrl}/${path}`, body, options).pipe(
      catchError(this.handleError)
    ) as Observable<T>;
  }

  delete<T>(path: string, options: any = {}): Observable<T> {
    return this.http.delete<T>(`${this.apiUrl}/${path}`, options).pipe(
      catchError(this.handleError)
    ) as Observable<T>;
  }

  private handleError(error: HttpErrorResponse) {
    if (error.status === 0) {
      console.error('An error occurred:', error.error);
    } else {
      console.error(
        `Backend returned code ${error.status}, body was: `, error.error);
    }
    return throwError(() => new Error('Something bad happened; please try again later.'));
  }
}
