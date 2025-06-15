import { ComponentFixture, TestBed } from '@angular/core/testing';

import { QaFormComponent } from './qa-form.component';

describe('QaFormComponent', () => {
  let component: QaFormComponent;
  let fixture: ComponentFixture<QaFormComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [QaFormComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(QaFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
