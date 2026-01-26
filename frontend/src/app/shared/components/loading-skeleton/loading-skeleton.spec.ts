import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LoadingSkeleton } from './loading-skeleton';

describe('LoadingSkeleton', () => {
  let component: LoadingSkeleton;
  let fixture: ComponentFixture<LoadingSkeleton>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LoadingSkeleton]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LoadingSkeleton);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
