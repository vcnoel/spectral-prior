# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from sklearn.utils.estimator_checks import parametrize_with_checks

from tabicl import TabICLClassifier


# use n_estimators=2 to test other preprocessing as well
@parametrize_with_checks([TabICLClassifier(n_estimators=2)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
