import { Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { AppConfig, DEFAULT_CONFIG } from './app.config';

@Injectable({
    providedIn: 'root'
})
export class ConfigService {
    private config = signal<AppConfig>(DEFAULT_CONFIG);
    private loaded = signal<boolean>(false);

    constructor(private http: HttpClient) { }

    /** Load configuration from JSON file */
    async loadConfig(): Promise<AppConfig> {
        try {
            const config = await this.http.get<Partial<AppConfig>>('/assets/config.json').toPromise();
            const merged = { ...DEFAULT_CONFIG, ...config };
            this.config.set(merged);
            this.loaded.set(true);
            return merged;
        } catch (error) {
            console.warn('Failed to load config.json, using defaults:', error);
            this.loaded.set(true);
            return DEFAULT_CONFIG;
        }
    }

    /** Get the current configuration */
    getConfig(): AppConfig {
        return this.config();
    }

    /** Check if config has been loaded */
    isLoaded(): boolean {
        return this.loaded();
    }

    /** Get a specific config value */
    get<K extends keyof AppConfig>(key: K): AppConfig[K] {
        return this.config()[key];
    }

    /** Check if a feature is enabled */
    isFeatureEnabled(feature: keyof AppConfig['features']): boolean {
        return this.config().features[feature];
    }

    /** Get navigation items */
    getNavigation(type: 'public' | 'admin') {
        return this.config().navigation[type];
    }
}
