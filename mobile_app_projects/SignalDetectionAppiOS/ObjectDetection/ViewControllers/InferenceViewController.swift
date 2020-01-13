// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import UIKit

// MARK: InferenceViewControllerDelegate Method Declarations
protocol InferenceViewControllerDelegate {

  /**
   This method is called when the user changes the stepper value to update number of threads used for inference.
   */
  func didChangeThreadCount(to count: Int)
}

class InferenceViewController: UIViewController {

  // MARK: Sections and Information to display
  private enum InferenceSections: Int, CaseIterable {
    case InferenceInfo
  }

  private enum InferenceInfo: Int, CaseIterable {
 
    case InferenceTime

    func displayString() -> String {

      var toReturn = ""

      switch self {
      case .InferenceTime:
        toReturn = "Inference Time"
      }
      return toReturn
    }
  }

  // MARK: Storyboard Outlets
  @IBOutlet weak var tableView: UITableView!
  @IBOutlet weak var threadStepper: UIStepper!
  @IBOutlet weak var stepperValueLabel: UILabel!

  // MARK: Constants
  private let normalCellHeight: CGFloat = 27.0
  private let separatorCellHeight: CGFloat = 42.0
  private let bottomSpacing: CGFloat = 21.0
  private let minThreadCount = 1
  private let bottomSheetButtonDisplayHeight: CGFloat = 60.0
  private let infoTextColor = UIColor.black
  private let lightTextInfoColor = UIColor(displayP3Red: 117.0/255.0, green: 117.0/255.0, blue: 117.0/255.0, alpha: 1.0)
  private let infoFont = UIFont.systemFont(ofSize: 14.0, weight: .regular)
  private let highlightedFont = UIFont.systemFont(ofSize: 14.0, weight: .medium)

  // MARK: Instance Variables
  var inferenceTime: Double = 0
  var wantedInputWidth: Int = 0
  var wantedInputHeight: Int = 0
  var resolution: CGSize = CGSize.zero
  var threadCountLimit: Int = 0
  var currentThreadCount: Int = 0

  // MARK: Delegate
  var delegate: InferenceViewControllerDelegate?

  // MARK: Computed properties
  var collapsedHeight: CGFloat {
    return bottomSheetButtonDisplayHeight

  }

  override func viewDidLoad() {
    super.viewDidLoad()

    // Set up stepper
    threadStepper.isUserInteractionEnabled = true
    threadStepper.maximumValue = Double(threadCountLimit)
    threadStepper.minimumValue = Double(minThreadCount)
    threadStepper.value = Double(currentThreadCount)

  }

  // MARK: Buttion Actions
  /**
   Delegate the change of number of threads to View Controller and change the stepper display.
   */
  @IBAction func onClickThreadStepper(_ sender: Any) {

    delegate?.didChangeThreadCount(to: Int(threadStepper.value))
    currentThreadCount = Int(threadStepper.value)
    stepperValueLabel.text = "\(currentThreadCount)"
  }
}


 
  // MARK: Format Display of information in the bottom sheet
  /*
   This method formats the display of additional information relating to the inferences.
   
  func displayStringsForInferenceInfo(atRow row: Int) -> (String, String) {

    var fieldName: String = ""
    var info: String = ""

    guard let inferenceInfo = InferenceInfo(rawValue: row) else {
      return (fieldName, info)
    }

    fieldName = inferenceInfo.displayString()

    switch inferenceInfo {

    case .InferenceTime:

      info = String(format: "%.2fms", inferenceTime)
    }

    return(fieldName, info)
  }
} */
